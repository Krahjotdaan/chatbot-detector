from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None) -> "BaseModel": ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]: ...

    def clone(self) -> "BaseModel":
        return self.__class__(**self._clone_kwargs())

    def _clone_kwargs(self) -> dict[str, Any]:
        return {}
    