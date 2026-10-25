import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def get_splits(y, groups, n_splits: int = 5, seed: int = 42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(y, y, groups))


def stratified_group_split(y, groups, val_size: float = 0.2, seed: int = 42):
    n_splits = max(2, round(1 / val_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return train_idx, val_idx