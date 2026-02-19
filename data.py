import numpy as np
import pandas as pd

def load_gz_rainfall_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # 日期列兼容：Unnamed: 0 / date
    if "date" in df.columns:
        date_col = "date"
    elif "Unnamed: 0" in df.columns:
        date_col = "Unnamed: 0"
    else:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # day-of-year sin/cos
    doy = df["date"].dt.dayofyear.values.astype(np.float32)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df

def split_by_year(df: pd.DataFrame, train_years, val_years, test_years):
    years = df["date"].dt.year.values
    train_mask = np.isin(years, train_years)
    val_mask = np.isin(years, val_years)
    test_mask = np.isin(years, test_years)
    return train_mask, val_mask, test_mask

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray):
        return x * self.std_ + self.mean_
