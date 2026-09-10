# Exercise 1: Import, Load, and View the Dataset
# Dataset: Campus Electricity ML Dataset

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np

# Step 2: Load the dataset
DATASET_FILE = "campus_electricity_weekly_experiment.csv"
df = pd.read_csv(DATASET_FILE)

# Step 3: View basic information about the dataset
print("=" * 60)
print("       CAMPUS ELECTRICITY ML DATASET - OVERVIEW")
print("=" * 60)
print(f"\nDataset File: {DATASET_FILE}")

# 3a. Shape of the dataset (rows, columns)
print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# 3b. Column names and data types
print("\n--- Column Names and Data Types ---")
print(df.dtypes)

# 3c. First 5 rows
print("\n--- First 5 Rows (head) ---")
print(df.head())

# 3d. Last 5 rows
print("\n--- Last 5 Rows (tail) ---")
print(df.tail())

# 3e. 100th row (pandas is 0-indexed, so index 99 is the 100th row)
print("\n--- 100th Row ---")
print(df.iloc[99:100])

# 3f. Random sample of 5 rows
print("\n--- Random Sample of 5 Rows ---")
print(df.sample(5, random_state=42))

# 3g. Dataset info (non-null counts and memory usage)
print("\n--- Dataset Info ---")
print(df.info())
