from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

TEXT_COL = "text"


def _split_text_numeric(X):
    if hasattr(X, "columns") and TEXT_COL in X.columns:
        return X[TEXT_COL], X.drop(columns=[TEXT_COL])
    return X, None


class LGBMTfidfModel(BaseModel):
    name = "lgbm_tfidf"

    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 50000,
        sublinear_tf: bool = True,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.9,
        class_weight: str | dict | None = "balanced",
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
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
        self._vec = None
        self._scaler = None
        self._clf = None
        self._feature_names: list[str] = []

    def _clone_kwargs(self) -> dict:
        return dict(
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

    def fit(self, X, y, X_val=None, y_val=None) -> "LGBMTfidfModel":
        text_tr, num_tr = _split_text_numeric(X)
        self._vec = self._make_vectorizer()
        Xtr = self._vec.fit_transform(text_tr)
        num_names: list[str] = []
        if num_tr is not None and num_tr.shape[1] > 0:
            self._scaler = StandardScaler().fit(num_tr)
            num_csr = sparse.csr_matrix(self._scaler.transform(num_tr))
            Xtr = sparse.hstack([Xtr, num_csr], format="csr")
            num_names = list(num_tr.columns)

        self._feature_names = [
            f"tfidf__{n}" for n in self._vec.get_feature_names_out()
        ] + [f"num__{n}" for n in num_names]

        y_arr = np.asarray(y)
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
            Xva = self._build_matrix(X_val)
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
        return self._clf.predict_proba(self._build_matrix(X))

    def get_params(self) -> dict:
        return self._clone_kwargs()

    def feature_importances(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        names = list(feature_names) if feature_names else self._feature_names
        return pd.DataFrame(
            {"feature": names, "importance": self._clf.feature_importances_}
        ).sort_values("importance", ascending=False)