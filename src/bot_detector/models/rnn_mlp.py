from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel
from .cnn_mlp import _build_vocab, _encode, _split_text_numeric

TEXT_COL = "text"


class HybridRnnMlpNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_num_features: int,
        rnn_type: str = "gru",
        embed_dim: int = 128,
        hidden_size: int = 64,
        num_layers: int = 1,
        hidden_hc: int = 8,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            embed_dim, hidden_size, num_layers=num_layers, batch_first=True
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
            nn.Linear(hidden_size + hidden_hc, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_hc, lengths):
        emb = self.embedding(x_seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        if isinstance(h, tuple):
            h = h[0]
        rnn_feat = h[-1]
        z = torch.cat([rnn_feat, self.mlp(x_hc)], dim=1)
        return self.head(z).squeeze(-1)


class RnnMlpModel(BaseModel):
    name = "rnn_mlp"

    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 64,
        num_layers: int = 1,
        max_len: int = 256,
        vocab_size: int = 20000,
        embed_dim: int = 128,
        hidden_hc: int = 8,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 3,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
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
            rnn_type=self.rnn_type,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            max_len=self.max_len,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
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
        seqs = torch.from_numpy(_encode(list(text), self._vocab, self.max_len))
        lengths = (seqs != 0).sum(dim=1).clamp(min=1)
        hc = torch.from_numpy(self._scaler.transform(num).astype(np.float32))
        return seqs, hc, lengths

    def fit(self, X, y, X_val=None, y_val=None) -> "RnnMlpModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"rnn_mlp device: {self._device}")

        text_tr, num_tr = _split_text_numeric(X)
        self._vocab = _build_vocab(text_tr, self.vocab_size)
        self._scaler = StandardScaler().fit(num_tr)

        seqs_tr, hc_tr, len_tr = self._tensors(X)
        y_tr = torch.tensor(np.asarray(y), dtype=torch.float32)

        n_pos = float((y_tr == 1).sum())
        n_neg = float((y_tr == 0).sum())
        pos_weight = None
        if self.class_weight == "balanced":
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self._device)

        self._net = HybridRnnMlpNet(
            vocab_size=len(self._vocab) + 2,
            n_num_features=hc_tr.shape[1],
            rnn_type=self.rnn_type,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            hidden_hc=self.hidden_hc,
        ).to(self._device)

        loader = DataLoader(
            TensorDataset(seqs_tr, hc_tr, len_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        has_val = X_val is not None and y_val is not None
        if has_val:
            seqs_va, hc_va, len_va = self._tensors(X_val)
            y_va = torch.tensor(np.asarray(y_val), dtype=torch.float32)

        best_state, best_loss, bad = None, np.inf, 0
        for epoch in range(self.epochs):
            self._net.train()
            for s, h, l, t in loader:
                s, h, l, t = (
                    s.to(self._device),
                    h.to(self._device),
                    l.to(self._device),
                    t.to(self._device),
                )
                opt.zero_grad()
                loss = loss_fn(self._net(s, h, l), t)
                loss.backward()
                opt.step()

            if not has_val:
                continue
            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    self._net(
                        seqs_va.to(self._device),
                        hc_va.to(self._device),
                        len_va.to(self._device),
                    ),
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
        seqs, hc, lengths = self._tensors(X)
        self._net.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                logits = self._net(
                    seqs[i : i + self.batch_size].to(self._device),
                    hc[i : i + self.batch_size].to(self._device),
                    lengths[i : i + self.batch_size].to(self._device),
                )
                probs.append(torch.sigmoid(logits).cpu().numpy())
        p = np.concatenate(probs)
        return np.stack([1 - p, p], axis=1)

    def get_params(self) -> dict:
        return self._clone_kwargs()