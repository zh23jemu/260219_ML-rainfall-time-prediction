import numpy as np
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    """
    返回：
    x: [L, V]
    y: [H]
    d: 域标签（int），若无则返回 -1
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, domain: np.ndarray, L: int, H: int):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.domain = domain.astype(np.int64) if domain is not None else None
        self.L = L
        self.H = H
        self.max_t = len(X) - H

    def __len__(self):
        return self.max_t - (self.L - 1)

    def __getitem__(self, idx):
        t = idx + (self.L - 1)
        x_win = self.X[t - self.L + 1 : t + 1]         # [L, V]
        y_fut = self.y[t + 1 : t + 1 + self.H]         # [H]
        d = -1 if self.domain is None else int(self.domain[t])
        return torch.from_numpy(x_win), torch.from_numpy(y_fut), torch.tensor(d, dtype=torch.long)
