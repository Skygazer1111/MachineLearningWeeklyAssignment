# Prepare a smaller weekly dataset for ML experiments
# Source dataset: campus_electricity_ml_dataset.csv

import pandas as pd


SOURCE_FILE = "campus_electricity_ml_dataset.csv"
OUTPUT_FILE = "campus_electricity_weekly_experiment.csv"
DAYS_TO_KEEP = 7

selected_columns = [
    "timestamp",
    "building_id",
    "building_type",
    "floor",
    "room_id",
    "room_area_m2",
    "capacity_students",
    "occupancy",
    "occupancy_rate",
    "temperature_c",
    "humidity_pct",
    "ac_usage_pct",
    "lighting_usage_pct",
    "plug_load_kw",
    "weekend",
    "holiday",
    "exam_period",
    "hour",
    "day_of_week",
    "peak_period",
    "power_kw",
    "energy_kwh_15min",
    "tariff_inr_per_kwh",
    "energy_cost_inr",
    "injected_anomaly",
]


print("Creating smaller weekly experiment dataset...")

# Load only useful columns to keep the experiment dataset focused.
df = pd.read_csv(SOURCE_FILE, usecols=selected_columns)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Keep the first complete week from the dataset.
start_date = df["timestamp"].min()
end_date = start_date + pd.Timedelta(days=DAYS_TO_KEEP)
df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]

# Keep one room per floor from each building.
selected_rooms = (
    df[["building_id", "floor", "room_id"]]
    .drop_duplicates()
    .sort_values(["building_id", "floor", "room_id"])
    .groupby(["building_id", "floor"])
    .head(1)["room_id"]
)

weekly_df = df[df["room_id"].isin(selected_rooms)].copy()
weekly_df = weekly_df.sort_values(["timestamp", "building_id", "room_id"])

weekly_df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("SMALL WEEKLY EXPERIMENT DATASET CREATED")
print("=" * 60)
print(f"Source rows kept      : {len(weekly_df):,}")
print(f"Columns kept          : {len(weekly_df.columns)}")
print(f"Buildings included    : {weekly_df['building_id'].nunique()}")
print(f"Rooms included        : {weekly_df['room_id'].nunique()}")
print(f"Date range            : {weekly_df['timestamp'].min()} to {weekly_df['timestamp'].max()}")
print(f"Output file           : {OUTPUT_FILE}")
