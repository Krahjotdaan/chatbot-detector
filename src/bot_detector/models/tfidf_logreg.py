import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

TEXT_COL = "text"


def _split_text_numeric(X):
    if hasattr(X, "columns") and TEXT_COL in X.columns:
        return X[TEXT_COL], X.drop(columns=[TEXT_COL])
    return X, None


class TfidfLogRegModel(BaseModel):
    name = "tfidf_logreg"

    def __init__(
        self,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (2, 5),
        max_features: int = 20000,
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.analyzer = analyzer
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self._vec = None
        self._scaler = None
        self._clf = None

    def _clone_kwargs(self) -> dict:
        return dict(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def _make_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )

    def _build_matrix(self, X):
        text, num = _split_text_numeric(X)
        Xv = self._vec.transform(text)
        if num is not None and num.shape[1] > 0 and self._scaler is not None:
            Xv = sparse.hstack(
                [Xv, sparse.csr_matrix(self._scaler.transform(num))], format="csr"
            )
        return Xv

    def fit(self, X, y, X_val=None, y_val=None) -> "TfidfLogRegModel":
        text_tr, num_tr = _split_text_numeric(X)
        self._vec = self._make_vectorizer()
        Xv = self._vec.fit_transform(text_tr)
        if num_tr is not None and num_tr.shape[1] > 0:
            self._scaler = StandardScaler().fit(num_tr)
            Xv = sparse.hstack(
                [Xv, sparse.csr_matrix(self._scaler.transform(num_tr))], format="csr"
            )
        self._clf = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(Xv, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(self._build_matrix(X))

    def get_params(self) -> dict:
        return self._clone_kwargs()