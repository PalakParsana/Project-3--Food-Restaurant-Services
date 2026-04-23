import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/restaurant_sales.csv")


def load_data(item: str = "Burger") -> pd.DataFrame:
    """
    Load the dataset, filter by menu item, and set a proper DatetimeIndex.

    """
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["item"] == item].copy()
    df = df.set_index("date").sort_index()

    # Ensure continuous daily index (no missing dates)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)

    # Fill any gaps
    df["sales"]               = df["sales"].ffill()
    df["is_weekend"]          = df["is_weekend"].fillna(0).astype(int)
    df["is_holiday"]          = df["is_holiday"].fillna(0).astype(int)
    df["is_festival"]         = df["is_festival"].fillna(0).astype(int)
    df["promo_active"]        = df["promo_active"].fillna(0).astype(int)
    df["temperature_celsius"] = df["temperature_celsius"].ffill()
    df["rainfall_mm"]         = df["rainfall_mm"].fillna(0)
    df["item"]                = df["item"].fillna(item)

    df.index.name = "date"
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str = "sales") -> pd.DataFrame:
    """Cap extreme outliers using the IQR fence method (caps, does not drop)."""
    Q1  = df[column].quantile(0.25)
    Q3  = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 3 * IQR
    lower_fence = max(0, Q1 - 3 * IQR)
    outliers    = ((df[column] > upper_fence) | (df[column] < lower_fence)).sum()
    df[column]  = df[column].clip(lower=lower_fence, upper=upper_fence)
    print(f"   Outliers capped: {outliers} rows")
    return df


def train_test_split_ts(df: pd.DataFrame,
                        test_months: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sequential train/test split — no shuffling, no data leakage.
    Last `test_months` months are held out as the test set.
    """
    split_date = df.index.max() - pd.DateOffset(months=test_months)
    train = df[df.index <= split_date].copy()
    test  = df[df.index >  split_date].copy()
    print(f"   Train: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} rows)")
    print(f"   Test : {test.index.min().date()}  → {test.index.max().date()}   ({len(test)} rows)")
    return train, test
