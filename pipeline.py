"""
pipeline.py
===========
Restaurant Demand Forecasting — Complete Data Pipeline

Steps:
1. Load data
2. Clean data
3. Create features
4. Train/Test split
5. Train model
6. Evaluate model
7. Save graphs
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# XGBoost if available
try:
    from xgboost import XGBRegressor
    USE_XGB = True
    print("XGBoost available")
except:
    USE_XGB = False
    print("XGBoost not found, using Random Forest")

# Settings
CSV_PATH = "Data/Restaurant_sales.csv"
ITEM_FILTER = "Burger"
TEST_MONTHS = 2
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 55)
print("   RESTAURANT DEMAND FORECASTING PIPELINE")
print("=" * 55)

# ---------------------------------------------------
# STEP 1 : LOAD DATA
# ---------------------------------------------------
print("\n[1/7] Loading data...")

df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

print("Rows   :", len(df))

print(f"      Total rows loaded : {len(df):,}")
print(f"      Columns           : {df.columns.tolist()}")
print(f"      Date range        : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"      Items             : {df['item'].unique().tolist()}")

# Filter Burger
item_df = df[df["item"] == ITEM_FILTER].copy()
item_df = item_df.sort_values("date").set_index("date")

# Fill missing dates
full_index = pd.date_range(item_df.index.min(), item_df.index.max(), freq="D")
item_df = item_df.reindex(full_index)

item_df["sales"] = item_df["sales"].ffill()
item_df["is_weekend"] = item_df["is_weekend"].fillna(0)
item_df["is_holiday"] = item_df["is_holiday"].fillna(0)
item_df["is_festival"] = item_df["is_festival"].fillna(0)
item_df["promo_active"] = item_df["promo_active"].fillna(0)
item_df["temperature_celsius"] = item_df["temperature_celsius"].ffill()
item_df["rainfall_mm"] = item_df["rainfall_mm"].fillna(0)

item_df.index.name = "date"

# ---------------------------------------------------
# STEP 2 : FEATURES
# ---------------------------------------------------
print("\n[2/7] Creating features...")

feat = item_df.copy()

# Calendar
feat["day_of_week"] = feat.index.dayofweek
feat["day_of_month"] = feat.index.day
feat["month"] = feat.index.month
feat["quarter"] = feat.index.quarter

# Lag
for lag in [1, 7, 14, 21, 28]:
    feat[f"lag_{lag}"] = feat["sales"].shift(lag)

# Rolling
for w in [7, 14, 30]:
    feat[f"rolling_mean_{w}"] = feat["sales"].shift(1).rolling(w).mean()

# Extra
feat["is_hot_day"] = (feat["temperature_celsius"] > 27).astype(int)
feat["is_rainy_day"] = (feat["rainfall_mm"] > 5).astype(int)

feat.dropna(inplace=True)

TARGET = "sales"
DROP_COLS = ["sales", "item"]

FEATURE_COLS = [c for c in feat.columns if c not in DROP_COLS]

X = feat[FEATURE_COLS]
y = feat[TARGET]

print("Features :", len(FEATURE_COLS))

# ---------------------------------------------------
# STEP 3 : SPLIT
# ---------------------------------------------------
print("\n[3/7] Train/Test split...")

split_date = feat.index.max() - pd.DateOffset(months=TEST_MONTHS)

X_train = X[X.index <= split_date]
X_test = X[X.index > split_date]

y_train = y[y.index <= split_date]
y_test = y[y.index > split_date]

print("Train rows :", len(X_train))
print("Test rows  :", len(X_test))

# ---------------------------------------------------
# STEP 4 : TRAIN MODEL
# ---------------------------------------------------
print("\n[4/7] Training model...")

if USE_XGB:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    model_name = "XGBoost"
else:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model_name = "Random Forest"

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model :", model_name)

# ---------------------------------------------------
# STEP 5 : EVALUATION
# ---------------------------------------------------
print("\n[5/7] Evaluating...")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"R2   : {r2:.4f}")

# ---------------------------------------------------
# STEP 6 : GRAPH
# ---------------------------------------------------
print("\n[6/7] Saving graph...")

plt.figure(figsize=(14,5))
plt.plot(y_test.index, y_test.values, label="Actual", color="steelblue")
plt.plot(y_test.index, y_pred, label="Predicted", color="coral", linestyle="--")

plt.fill_between(
    y_test.index,
    y_pred - mae,
    y_pred + mae,
    alpha=0.15,
    color="coral"
)

plt.title("Forecast vs Actual")
plt.legend()
plt.tight_layout()

plt.savefig("reports/forecast_chart.png")
plt.show()

# ---------------------------------------------------
# STEP 7 : FEATURE IMPORTANCE
# ---------------------------------------------------
print("\n[7/7] Feature importance...")

feat_imp = pd.Series(
    model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False).head(15)

plt.figure(figsize=(10,7))
plt.barh(feat_imp.index[::-1], feat_imp.values[::-1], color="steelblue")
plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("reports/feature_importance.png")
plt.show()

print("\nTop Features:")
print(feat_imp.head())

# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------
print("\n" + "=" * 55)
print("BUSINESS SUMMARY")
print("=" * 55)

print("Item Forecasted :", ITEM_FILTER)
print("Model Used      :", model_name)
print("Avg Sales       :", round(y_test.mean()))
print("Forecast Error  :", round(mae))
print("Accuracy        :", round(100 - mape, 1), "%")

print("=" * 55)
print("Pipeline complete")
print("Files saved in reports folder")