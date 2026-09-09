# Exercise 2: Display Summary and Statistics of the Dataset
# Dataset: Campus Electricity ML Dataset

import pandas as pd

# Load the dataset
df = pd.read_csv("campus_electricity_ml_dataset.csv")

print("=" * 55)
print("  CAMPUS ELECTRICITY DATASET - SUMMARY & STATISTICS")
print("=" * 55)

# --- Dataset Overview ---
print("\n>> DATASET OVERVIEW")
print(f"  Total Records      : {df.shape[0]:,}")
print(f"  Total Features     : {df.shape[1]}")
print(f"  Buildings           : {df['building_id'].nunique()}")
print(f"  Building Types      : {', '.join(df['building_type'].unique())}")
print(f"  Total Rooms         : {df['room_id'].nunique()}")
print(f"  Date Range          : {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Missing Values      : {df.isnull().sum().sum():,} ({(df.isnull().sum().sum()/df.size*100):.2f}%)")

# --- Electricity Usage Statistics ---
print("\n>> ELECTRICITY USAGE STATISTICS")
print(f"  Avg Power Consumption     : {df['power_kw'].mean():.2f} kW")
print(f"  Median Power Consumption  : {df['power_kw'].median():.2f} kW")
print(f"  Max Power Consumption     : {df['power_kw'].max():.2f} kW")
print(f"  Min Power Consumption     : {df['power_kw'].min():.2f} kW")
print(f"  Avg Energy per 15 min     : {df['energy_kwh_15min'].mean():.2f} kWh")
print(f"  Total Energy Consumed     : {df['energy_kwh_15min'].sum():,.2f} kWh")

# --- Cost Statistics ---
print("\n>> ENERGY COST STATISTICS")
print(f"  Avg Cost per 15 min       : Rs.{df['energy_cost_inr'].mean():.2f}")
print(f"  Median Cost per 15 min    : Rs.{df['energy_cost_inr'].median():.2f}")
print(f"  Max Cost (single reading) : Rs.{df['energy_cost_inr'].max():.2f}")
print(f"  Total Energy Cost         : Rs.{df['energy_cost_inr'].sum():,.2f}")
print(f"  Tariff Rates              : Rs.{df['tariff_inr_per_kwh'].unique()} per kWh")

# --- Building-wise Comparison ---
print("\n>> AVERAGE USAGE BY BUILDING TYPE")
building_stats = df.groupby("building_type").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean"),
    Avg_Occupancy=("occupancy", "mean")
).round(2)
print(building_stats.to_string())

# --- Peak vs Off-Peak ---
print("\n>> PEAK vs OFF-PEAK HOURS")
peak = df.groupby("peak_period").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean")
).round(2)
peak.index = ["Off-Peak", "Peak"]
print(peak.to_string())

# --- Weekday vs Weekend ---
print("\n>> WEEKDAY vs WEEKEND")
weekend = df.groupby("weekend").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean"),
    Avg_Occupancy=("occupancy", "mean")
).round(2)
weekend.index = ["Weekday", "Weekend"]
print(weekend.to_string())

# --- Monthly Trend ---
print("\n>> MONTHLY ENERGY USAGE TREND")
monthly = df.groupby("month").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean"),
    Avg_Temp_C=("temperature_c", "mean"),
    Avg_AC_Usage_pct=("ac_usage_pct", "mean")
).round(2)
month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June"}
monthly.index = monthly.index.map(month_names)
print(monthly.to_string())

# --- Time of Day Analysis ---
print("\n>> TIME OF DAY ANALYSIS")
def get_time_slot(hour):
    if hour < 6: return "Night (12am-6am)"
    elif hour < 12: return "Morning (6am-12pm)"
    elif hour < 18: return "Afternoon (12pm-6pm)"
    else: return "Evening (6pm-12am)"
df["time_slot"] = df["hour"].apply(get_time_slot)
time_stats = df.groupby("time_slot").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean"),
    Avg_Occupancy=("occupancy", "mean")
).round(2)
print(time_stats.to_string())

# --- Holiday vs Non-Holiday ---
print("\n>> HOLIDAY vs NON-HOLIDAY")
holiday = df.groupby("holiday").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean"),
    Avg_Occupancy=("occupancy", "mean")
).round(2)
holiday.index = ["Regular Day", "Holiday"]
print(holiday.to_string())

# --- Exam Period vs Normal Period ---
print("\n>> EXAM PERIOD vs NORMAL PERIOD")
exam = df.groupby("exam_period").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean"),
    Avg_Occupancy=("occupancy", "mean")
).round(2)
exam.index = ["Normal Period", "Exam Period"]
print(exam.to_string())

# --- Floor-wise Comparison ---
print("\n>> FLOOR-WISE ENERGY USAGE")
floor_stats = df.groupby("floor").agg(
    Avg_Power_kW=("power_kw", "mean"),
    Avg_Cost_INR=("energy_cost_inr", "mean"),
    Avg_Occupancy=("occupancy", "mean")
).round(2)
floor_stats.index = [f"Floor {f}" for f in floor_stats.index]
print(floor_stats.to_string())

# --- AC Usage vs Temperature ---
print("\n>> AC USAGE vs TEMPERATURE (Correlation)")
corr = df[["temperature_c", "ac_usage_pct", "power_kw"]].corr().round(3)
print(corr.to_string())
print(f"\n  Higher temperature -> Higher AC usage -> Higher power consumption")

# --- Environmental Conditions ---
print("\n>> ENVIRONMENTAL CONDITIONS (Average)")
print(f"  Temperature        : {df['temperature_c'].mean():.2f} C")
print(f"  Humidity           : {df['humidity_pct'].mean():.2f} %")
print(f"  Solar Irradiance   : {df['solar_irradiance_w_m2'].mean():.2f} W/m2")

# --- Anomalies ---
anomaly_count = df['injected_anomaly'].sum()
print(f"\n>> ANOMALIES DETECTED: {anomaly_count:,} out of {len(df):,} records ({anomaly_count/len(df)*100:.2f}%)")

# Cleanup temporary column
df.drop(columns=["time_slot"], inplace=True)
