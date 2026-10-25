from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..features import tokenize
from .base import BaseModel

TEXT_COL = "text"


def _split_text_numeric(X):
    if hasattr(X, "columns") and TEXT_COL in X.columns:
        return X[TEXT_COL], X.drop(columns=[TEXT_COL])
    return X, None


def _build_vocab(texts, vocab_size):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {}
    for tok, cnt in counter.most_common(vocab_size - 2):
        if cnt < 2:
            break
        vocab[tok] = len(vocab) + 2
    return vocab


def _encode(texts, vocab, max_len):
    seqs = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(tok, 1) for tok in tokenize(t)[:max_len]]
        seqs[i, : len(ids)] = ids
    return seqs


class HybridCnnMlpNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_num_features: int,
        embed_dim: int = 128,
        kernel_sizes=(2, 3, 4, 5),
        num_filters: int = 64,
        hidden_cnn: int = 64,
        hidden_hc: int = 8,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embed_dim, num_filters, k),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                for k in kernel_sizes
            ]
        )
        self.cnn_out = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), hidden_cnn), nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, hidden_hc),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_cnn + hidden_hc, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_hc):
        emb = self.embedding(x_seq).permute(0, 2, 1)
        cnn = torch.cat([conv(emb).squeeze(-1) for conv in self.convs], dim=1)
        z = torch.cat([self.cnn_out(cnn), self.mlp(x_hc)], dim=1)
        return self.head(z).squeeze(-1)


class CnnMlpModel(BaseModel):
    name = "cnn_mlp"

    def __init__(
        self,
        max_len: int = 128,
        vocab_size: int = 20000,
        embed_dim: int = 128,
        kernel_sizes=(2, 3, 4, 5),
        num_filters: int = 64,
        hidden_cnn: int = 64,
        hidden_hc: int = 8,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 3,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.kernel_sizes = list(kernel_sizes)
        self.num_filters = num_filters
        self.hidden_cnn = hidden_cnn
        self.hidden_hc = hidden_hc
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.class_weight = class_weight
        self.random_state = random_state
        self._vocab = None
        self._scaler = None
        self._net = None
        self._device = None

    def _clone_kwargs(self) -> dict:
        return dict(
            max_len=self.max_len,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            kernel_sizes=self.kernel_sizes,
            num_filters=self.num_filters,
            hidden_cnn=self.hidden_cnn,
            hidden_hc=self.hidden_hc,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def _tensors(self, X):
        text, num = _split_text_numeric(X)
        if num is None or num.shape[1] == 0 or self._scaler is None:
            raise ValueError("cnn_mlp требует числовые фичи вместе с текстом")
        seqs = torch.from_numpy(_encode(list(text), self._vocab, self.max_len))
        hc = torch.from_numpy(self._scaler.transform(num).astype(np.float32))
        return seqs, hc

    def fit(self, X, y, X_val=None, y_val=None) -> "CnnMlpModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"cnn_mlp device: {self._device}")

        text_tr, num_tr = _split_text_numeric(X)
        self._vocab = _build_vocab(text_tr, self.vocab_size)
        self._scaler = StandardScaler().fit(num_tr)

        seqs_tr, hc_tr = self._tensors(X)
        y_tr = torch.tensor(np.asarray(y), dtype=torch.float32)

        n_pos = float((y_tr == 1).sum())
        n_neg = float((y_tr == 0).sum())
        pos_weight = None
        if self.class_weight == "balanced":
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self._device)

        self._net = HybridCnnMlpNet(
            vocab_size=len(self._vocab) + 2,
            n_num_features=hc_tr.shape[1],
            embed_dim=self.embed_dim,
            kernel_sizes=tuple(self.kernel_sizes),
            num_filters=self.num_filters,
            hidden_cnn=self.hidden_cnn,
            hidden_hc=self.hidden_hc,
        ).to(self._device)

        loader = DataLoader(
            TensorDataset(seqs_tr, hc_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        has_val = X_val is not None and y_val is not None
        if has_val:
            seqs_va, hc_va = self._tensors(X_val)
            y_va = torch.tensor(np.asarray(y_val), dtype=torch.float32)

        best_state, best_loss, bad = None, np.inf, 0
        for epoch in range(self.epochs):
            self._net.train()
            for s, h, t in loader:
                s, h, t = s.to(self._device), h.to(self._device), t.to(self._device)
                opt.zero_grad()
                loss = loss_fn(self._net(s, h), t)
                loss.backward()
                opt.step()

            if not has_val:
                continue
            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    self._net(seqs_va.to(self._device), hc_va.to(self._device)),
                    y_va.to(self._device),
                ).item()
            if val_loss < best_loss - 1e-4:
                best_loss, bad = val_loss, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._net.state_dict().items()
                }
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        seqs, hc = self._tensors(X)
        self._net.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                logits = self._net(
                    seqs[i : i + self.batch_size].to(self._device),
                    hc[i : i + self.batch_size].to(self._device),
                )
                probs.append(torch.sigmoid(logits).cpu().numpy())
        p = np.concatenate(probs)
        return np.stack([1 - p, p], axis=1)

    def get_params(self) -> dict:
        return self._clone_kwargs()