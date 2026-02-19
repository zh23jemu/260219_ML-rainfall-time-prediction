import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred=np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred=np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def picp(y_true, y_lo, y_hi):
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))

def pinaw(y_lo, y_hi, y_true):
    y_true = np.asarray(y_true)
    denom = np.max(y_true) - np.min(y_true) + 1e-6
    return float(np.mean(y_hi - y_lo) / denom)

def aql(y_true, y_lo, y_hi, alpha=0.1):
    # Average Quantile Loss for interval [alpha/2, 1-alpha/2]
    y_true=np.asarray(y_true)
    below = np.maximum(y_lo - y_true, 0)
    above = np.maximum(y_true - y_hi, 0)
    return float(np.mean(below + above))
