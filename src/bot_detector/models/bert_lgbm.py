import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .base import BaseModel
from .cnn_mlp import _split_text_numeric


class BertBackbone(nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean") -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def _pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.head(self.dropout(pooled)).squeeze(-1), pooled

    def embed(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    def freeze_first_layers(self, n: int) -> None:
        if n <= 0:
            return
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for layer in self.backbone.encoder.layer[:n]:
            for p in layer.parameters():
                p.requires_grad = False


class BertLGBMModel(BaseModel):
    name = "bert_lgbm"

    def __init__(
        self,
        bert_model_name: str = "cointegrated/rubert-tiny2",
        max_len: int = 256,
        bert_epochs: int = 4,
        bert_batch_size: int = 16,
        bert_lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        grad_clip: float = 1.0,
        bert_patience: int = 2,
        pooling: str = "mean",
        freeze_layers: int = 0,
        bert_class_weight: str | dict | None = "balanced",
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 50000,
        sublinear_tf: bool = True,
        n_estimators: int = 2000,
        learning_rate: float = 0.03,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        class_weight: str | dict | None = "balanced",
        early_stopping_rounds: int = 100,
        random_state: int = 42,
    ) -> None:
        self.bert_model_name = bert_model_name
        self.max_len = max_len
        self.bert_epochs = bert_epochs
        self.bert_batch_size = bert_batch_size
        self.bert_lr = bert_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip = grad_clip
        self.bert_patience = bert_patience
        self.pooling = pooling
        self.freeze_layers = freeze_layers
        self.bert_class_weight = bert_class_weight
        self.analyzer = analyzer
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.class_weight = class_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._tokenizer = None
        self._bert: BertBackbone | None = None
        self._vec = None
        self._scaler = None
        self._clf = None
        self._device = None
        self._feature_names: list[str] = []

    def _clone_kwargs(self) -> dict:
        return dict(
            bert_model_name=self.bert_model_name,
            max_len=self.max_len,
            bert_epochs=self.bert_epochs,
            bert_batch_size=self.bert_batch_size,
            bert_lr=self.bert_lr,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            grad_clip=self.grad_clip,
            bert_patience=self.bert_patience,
            pooling=self.pooling,
            freeze_layers=self.freeze_layers,
            bert_class_weight=self.bert_class_weight,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            class_weight=self.class_weight,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
        )


    def _tokenize(self, texts) -> dict[str, torch.Tensor]:
        return self._tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

    def _make_loader(self, texts, y: np.ndarray, shuffle: bool) -> DataLoader:
        tok = self._tokenize(texts)
        dataset = TensorDataset(
            tok["input_ids"],
            tok["attention_mask"],
            torch.tensor(np.asarray(y), dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            batch_size=self.bert_batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.random_state),
        )

    def _embed(self, texts) -> np.ndarray:
        tok = self._tokenize(texts)
        chunks = []
        bs = self.bert_batch_size
        self._bert.eval()
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                emb = self._bert.embed(
                    tok["input_ids"][i : i + bs].to(self._device),
                    tok["attention_mask"][i : i + bs].to(self._device),
                )
                chunks.append(emb.cpu().numpy())
        return np.concatenate(chunks)

    def _val_loss(self, X_val, y_val, loss_fn) -> float:
        text_va, _ = _split_text_numeric(X_val)
        loader = self._make_loader(text_va, np.asarray(y_val), shuffle=False)
        self._bert.eval()
        losses = []
        with torch.no_grad():
            for input_ids, attention_mask, targets in loader:
                logits, _ = self._bert(
                    input_ids.to(self._device), attention_mask.to(self._device)
                )
                losses.append(
                    loss_fn(logits, targets.to(self._device)).item() * len(targets)
                )
        return float(np.sum(losses) / len(loader.dataset))

    def _fine_tune(self, text_tr, y_arr: np.ndarray, X_val, y_val) -> None:
        self._bert = BertBackbone(self.bert_model_name, self.pooling).to(self._device)
        if self.bert_epochs <= 0:
            print("  bert: frozen (bert_epochs=0), skip fine-tune")
            self._bert.eval()
            return
        self._bert.freeze_first_layers(self.freeze_layers)

        pos_weight = None
        if self.bert_class_weight == "balanced":
            n_pos = float((y_arr == 1).sum())
            n_neg = float((y_arr == 0).sum())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self._device)

        loader = self._make_loader(text_tr, y_arr, shuffle=True)
        total_steps = max(1, len(loader) * self.bert_epochs)

        no_decay = ("bias", "LayerNorm.weight")
        trainable = [(n, p) for n, p in self._bert.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for n, p in trainable if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p for n, p in trainable if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=self.bert_lr,
        )
        sched = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )

        has_val = X_val is not None and y_val is not None
        best_state, best_loss, bad = None, np.inf, 0
        for epoch in range(self.bert_epochs):
            self._bert.train()
            for input_ids, attention_mask, targets in loader:
                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                targets = targets.to(self._device)
                opt.zero_grad()
                logits, _ = self._bert(input_ids, attention_mask)
                loss = loss_fn(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self._bert.parameters(), self.grad_clip)
                opt.step()
                sched.step()

            if not has_val:
                continue
            val_loss = self._val_loss(X_val, y_val, loss_fn)
            print(f"  bert epoch {epoch}: val_loss={val_loss:.4f}")
            if val_loss < best_loss - 1e-4:
                best_loss, bad = val_loss, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._bert.state_dict().items()
                }
            else:
                bad += 1
                if bad >= self.bert_patience:
                    break

        if best_state is not None:
            self._bert.load_state_dict(best_state)
        self._bert.eval()


    def _combine(self, emb: np.ndarray, X_tfidf, num) -> sparse.csr_matrix:
        blocks = [sparse.csr_matrix(emb.astype(np.float32)), X_tfidf]
        if num is not None and num.shape[1] > 0 and self._scaler is not None:
            blocks.append(sparse.csr_matrix(self._scaler.transform(num)))
        return sparse.hstack(blocks, format="csr")


    def fit(self, X, y, X_val=None, y_val=None) -> "BertLGBMModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"bert_lgbm device: {self._device}")

        text_tr, num_tr = _split_text_numeric(X)
        self._tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)

        y_arr = np.asarray(y)
        self._fine_tune(text_tr, y_arr, X_val, y_val)

        self._vec = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        Xtr_tfidf = self._vec.fit_transform(text_tr)
        num_names: list[str] = []
        if num_tr is not None and num_tr.shape[1] > 0:
            self._scaler = StandardScaler().fit(num_tr)
            num_names = list(num_tr.columns)
        else:
            self._scaler = None

        emb_tr = self._embed(text_tr)
        Xtr = self._combine(emb_tr, Xtr_tfidf, num_tr)
        self._feature_names = (
            [f"bert__emb_{i}" for i in range(emb_tr.shape[1])]
            + [f"tfidf__{n}" for n in self._vec.get_feature_names_out()]
            + [f"num__{n}" for n in num_names]
        )

        clf = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            class_weight=self.class_weight,
            random_state=self.random_state,
            objective="binary",
            metric="binary_logloss",
            verbosity=-1,
        )

        if X_val is not None and y_val is not None:
            text_va, num_va = _split_text_numeric(X_val)
            emb_va = self._embed(text_va)
            Xva = self._combine(emb_va, self._vec.transform(text_va), num_va)
            clf.fit(
                Xtr,
                y_arr,
                eval_X=Xva,
                eval_y=np.asarray(y_val),
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        else:
            clf.fit(Xtr, y_arr, callbacks=[lgb.log_evaluation(0)])

        self._clf = clf
        return self

    def predict_proba(self, X) -> np.ndarray:
        text, num = _split_text_numeric(X)
        emb = self._embed(text)
        Xm = self._combine(emb, self._vec.transform(text), num)
        return self._clf.predict_proba(Xm)

    def get_params(self) -> dict:
        return self._clone_kwargs()

    def feature_importances(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        names = list(feature_names) if feature_names else self._feature_names
        return pd.DataFrame(
            {"feature": names, "importance": self._clf.feature_importances_}
        ).sort_values("importance", ascending=False)
