import numpy as np
from sklearn.mixture import GaussianMixture

def compute_window_stats(X: np.ndarray, y: np.ndarray, L: int, H: int):
    """
    计算每个时间点 t 对应窗口的统计特征 s_t（用于 GMM）。
    注意：这里的特征使用 X/y（可用标准化后的或原始的均可）。
    返回：
      S: [T, d_s]，只有 t>=L-1 且 t<=T-H-1 才有效，其余位置为 nan
    """
    T, V = X.shape
    S = np.full((T, 8), np.nan, dtype=np.float32)

    for t in range(L - 1, T - H):
        yw = y[t - L + 1 : t + 1]
        # 基础统计：均值、方差、湿日比例、极端分位数
        mean = yw.mean()
        std = yw.std()
        wet = (yw > 0).mean()
        q90 = np.quantile(yw, 0.90)
        q99 = np.quantile(yw, 0.99)
        # 邻域综合强度（X 的均值/方差）
        xw = X[t - L + 1 : t + 1]
        xmean = xw.mean()
        xstd = xw.std()
        # 目标与邻域平均相关（粗略）
        corr = 0.0
        if V > 1:
            yc = yw - yw.mean()
            denom_y = (yc**2).sum() + 1e-6
            corrs = []
            for j in range(V):
                xj = xw[:, j] - xw[:, j].mean()
                denom_x = (xj**2).sum() + 1e-6
                corrs.append(float((yc * xj).sum() / np.sqrt(denom_y * denom_x)))
            corr = float(np.mean(corrs))
        S[t] = [mean, std, wet, q90, q99, xmean, xstd, corr]

    return S

def fit_gmm_bic(S_train: np.ndarray, k_min=2, k_max=6, random_state=42):
    S_train = S_train[~np.isnan(S_train).any(axis=1)]
    best_gmm = None
    best_bic = 1e18
    best_k = None

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gmm.fit(S_train)
        bic = gmm.bic(S_train)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    return best_gmm, best_k, best_bic

def predict_domain(gmm: GaussianMixture, S: np.ndarray):
    dom = np.full((len(S),), -1, dtype=np.int64)
    valid = ~np.isnan(S).any(axis=1)
    dom[valid] = gmm.predict(S[valid]).astype(np.int64)
    return dom
