import pandas as pd
import numpy as np
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE  = "2022-01-01"
END_DATE    = "2023-12-31"
MENU_ITEMS  = ["Burger", "Pizza", "Pasta", "Salad"]
OUTPUT_PATH = Path("data/restaurant_sales.csv")

# ── Indian Public Holidays ────────────────────────────────────────────────────
HOLIDAYS = [
    "2022-01-26", "2022-03-18", "2022-04-14", "2022-08-15",
    "2022-10-02", "2022-10-05", "2022-10-24", "2022-11-08", "2022-12-25",
    "2023-01-26", "2023-03-08", "2023-04-14", "2023-08-15",
    "2023-10-02", "2023-10-24", "2023-11-27", "2023-12-25",
]

# ── Indian Festival Dates (high footfall events) ──────────────────────────────
FESTIVALS = [
    # Diwali
    "2022-10-24", "2022-10-25", "2022-10-26",
    "2023-11-12", "2023-11-13", "2023-11-14",
    # Holi
    "2022-03-17", "2022-03-18",
    "2023-03-07", "2023-03-08",
    # Navratri / Dussehra
    "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05",
    "2023-10-15", "2023-10-16", "2023-10-24",
    # New Year
    "2022-01-01", "2023-01-01",
    # Christmas
    "2022-12-24", "2022-12-25",
    "2023-12-24", "2023-12-25",
    # Eid
    "2022-05-02", "2022-05-03",
    "2023-04-21", "2023-04-22",
]

# ── Promo campaign windows (date ranges when discounts/combos run) ────────────
PROMO_WINDOWS = [
    ("2022-02-10", "2022-02-16"),   # Valentine's week combo
    ("2022-05-01", "2022-05-07"),   # Summer launch promo
    ("2022-08-10", "2022-08-16"),   # Independence Day offer
    ("2022-11-01", "2022-11-07"),   # Post-Diwali clearance
    ("2022-12-20", "2022-12-31"),   # Year-end sale
    ("2023-02-10", "2023-02-16"),   # Valentine's week combo
    ("2023-06-01", "2023-06-07"),   # Monsoon special
    ("2023-08-10", "2023-08-16"),   # Independence Day offer
    ("2023-11-15", "2023-11-21"),   # Post-Diwali clearance
    ("2023-12-20", "2023-12-31"),   # Year-end sale
]

# ── Bengaluru monthly avg temperature (°C) ────────────────────────────────────
MONTHLY_TEMP = {
    1: 21.0, 2: 23.5, 3: 26.5, 4: 28.0,
    5: 27.5, 6: 24.0, 7: 22.5, 8: 22.5,
    9: 23.0, 10: 23.0, 11: 21.5, 12: 20.0,
}

# ── Bengaluru monthly avg rainfall (mm/day) ───────────────────────────────────
MONTHLY_RAIN = {
    1: 0.2, 2: 0.3, 3: 0.5, 4: 1.5,
    5: 3.5, 6: 7.0, 7: 9.5, 8: 8.0,
    9: 8.5, 10: 7.0, 11: 3.0, 12: 0.5,
}

# ── Base daily demand per item (weekday baseline) ─────────────────────────────
BASE_DEMAND = {
    "Burger": 80,
    "Pizza":  65,
    "Pasta":  50,
    "Salad":  35,
}


def build_promo_set():
    promo_dates = set()
    for start, end in PROMO_WINDOWS:
        for d in pd.date_range(start, end, freq="D"):
            promo_dates.add(d)
    return promo_dates


def generate_sales_data() -> pd.DataFrame:
    date_range   = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    holidays     = pd.to_datetime(HOLIDAYS)
    festivals    = pd.to_datetime(FESTIVALS)
    promo_dates  = build_promo_set()
    records      = []

    for date in date_range:
        # ── Shared date-level signals ─────────────────────────────────────────
        is_holiday  = int(date in holidays)
        is_festival = int(date in festivals)
        is_promo    = int(date in promo_dates)
        is_weekend  = int(date.dayofweek >= 5)

        # Temperature: monthly avg ± small daily noise
        temp = round(MONTHLY_TEMP[date.month] + np.random.normal(0, 1.2), 1)

        # Rainfall: monthly avg × random factor (many zero-rain days)
        rain_mean = MONTHLY_RAIN[date.month]
        rainfall  = round(max(0, np.random.exponential(rain_mean)), 1)

        for item in MENU_ITEMS:
            base = BASE_DEMAND[item]

            # 1. Growth trend (~20% over 2 years)
            days_elapsed = (date - pd.Timestamp(START_DATE)).days
            trend        = 1 + (0.20 * days_elapsed / 730)

            # 2. Weekly seasonality
            weekly = {5: 1.45, 6: 1.35, 4: 1.15}.get(date.dayofweek, 1.0)

            # 3. Monthly seasonality
            monthly = {
                1: 1.10, 2: 0.95, 3: 0.95, 4: 1.00,
                5: 1.05, 6: 1.00, 7: 0.90, 8: 0.90,
                9: 1.00, 10: 1.20, 11: 1.10, 12: 1.30
            }[date.month]

            # 4. Holiday spike
            holiday_mult = 1.60 if is_holiday else 1.0

            # 5. Festival spike (bigger than holiday for food business)
            festival_mult = 1.80 if is_festival else 1.0

            # 6. Promo boost
            promo_mult = 1.25 if is_promo else 1.0

            # 7. Temperature effect (item-specific)
            # Salad demand rises in heat; Soup/Pasta dips slightly
            if item == "Salad":
                temp_mult = 1 + 0.015 * (temp - 24)   # warmer → more salad
            elif item == "Pasta":
                temp_mult = 1 - 0.008 * (temp - 24)   # warmer → slightly less pasta
            else:
                temp_mult = 1.0                         # Burger/Pizza neutral

            # 8. Rainfall effect (rain reduces footfall for all items)
            rain_mult = max(0.70, 1 - 0.03 * rainfall)

            # 9. Random noise (±12%)
            noise = np.random.normal(1.0, 0.10)

            sales = max(0, int(
                base * trend * weekly * monthly
                * holiday_mult * festival_mult
                * promo_mult * temp_mult * rain_mult
                * noise
            ))

            records.append({
                "date"               : date,
                "item"               : item,
                "sales"              : sales,
                "is_weekend"         : is_weekend,
                "is_holiday"         : is_holiday,
                "is_festival"        : is_festival,
                "promo_active"       : is_promo,
                "temperature_celsius": temp,
                "rainfall_mm"        : rainfall,
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"  Dataset saved to '{OUTPUT_PATH}'")
    print(f"   Rows         : {len(df):,}")
    print(f"   Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Items        : {df['item'].unique().tolist()}")
    print(f"   Columns      : {df.columns.tolist()}")
    print(f"   Festival rows: {df['is_festival'].sum()}")
    print(f"   Promo rows   : {df['promo_active'].sum()}")
    return df


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_sales_data()
