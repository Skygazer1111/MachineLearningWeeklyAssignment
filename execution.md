# Project Execution Guide

This file contains the commands needed to run each part of the Machine Learning weekly experiment project.

## 1. Install Required Libraries

Run this command once before executing the programs:

```bash
pip install pandas numpy scikit-learn
```

## 2. Prepare the Smaller Weekly Experiment Dataset

The original dataset is very large, so this command creates a smaller dataset for weekly ML experiments.

```bash
python prepare_weekly_experiment_dataset.py
```

Expected output:

```text
Creating smaller weekly experiment dataset...
============================================================
SMALL WEEKLY EXPERIMENT DATASET CREATED
============================================================
Source rows kept      : 13,440
Columns kept          : 25
Buildings included    : 5
Rooms included        : 20
Date range            : 2025-01-01 00:00:00 to 2025-01-07 23:45:00
Output file           : campus_electricity_weekly_experiment.csv
```

## 3. Exercise 1: Import, Load, and View Dataset

Run:

```bash
python exercise1_load_view.py
```

This program displays:

```text
Dataset File: campus_electricity_weekly_experiment.csv
Dataset Shape: 13440 rows x 25 columns
Column names and data types
First 5 rows
Last 5 rows
100th row
Random sample rows
Dataset info
```

View the 100th row the same way as `head` and `tail`. Pandas uses 0-based indexing, so the 100th row is at index 99:

```python
df.head()          # first 5 rows
df.tail()          # last 5 rows
df.iloc[99:100]    # 100th row
df.sample(5)       # 5 random rows
```

## 4. Exercise 2: Summary and Statistics

Run:

```bash
python exercise2_summary_statistics.py
```

This program displays important statistics such as:

```text
Total Records          : 13,440
Total Features         : 25
Buildings              : 5
Total Rooms            : 20
Average Power Usage    : 7.34 kW
Total Energy Consumed  : 24,656.47 kWh
Average Cost           : Rs.11.08 per 15 min
Total Energy Cost      : Rs.148,974.70
Anomalies Detected     : 53 records
```

It also compares:

```text
Building type usage
Peak vs off-peak usage
Weekday vs weekend usage
Day-wise energy usage
Time-of-day energy usage
Holiday vs regular day usage
Floor-wise energy usage
AC usage vs temperature correlation
```

## 5. Exercise 3: Linear Regression Prediction

Run:

```bash
python exercise3_linear_regression.py
```

This program trains a Linear Regression model to predict electricity usage (`power_kw`).

Expected output summary:

```text
Dataset File        : campus_electricity_weekly_experiment.csv
Total Records       : 13,440
Training Records    : 10,752
Testing Records     : 2,688
Target Variable     : power_kw
Number of Features  : 18

Mean Absolute Error : 0.45 kW
Root Mean Sq. Error : 1.01 kW
R2 Score            : 0.758
Result              : Strong prediction performance
```

The program also displays sample actual vs predicted electricity usage values.

## 6. Exercise 4: Bayesian Logistic Regression and SVM Classification

Run:

```bash
python exercise4_classification_models.py
```

This program classifies electricity readings as:

```text
0 = Normal reading
1 = Anomalous reading
```

Expected output summary:

```text
Dataset File        : campus_electricity_weekly_experiment.csv
Total Records       : 13,440
Training Records    : 10,752
Testing Records     : 2,688
Normal readings     : 13,387
Anomaly readings    : 53
Anomaly percentage  : 0.39%
```

Bayesian Logistic Regression result:

```text
Accuracy        : 0.999
Anomaly recall  : 0.91
Detected        : 10 out of 11 anomalies in the test set
```

SVM result:

```text
Accuracy        : 0.999
Anomaly precision: 0.91
Anomaly recall  : 0.91
Detected        : 10 out of 11 anomalies in the test set
```

## 7. Recommended Execution Order

Run the programs in this order:

```bash
python prepare_weekly_experiment_dataset.py
python exercise1_load_view.py
python exercise2_summary_statistics.py
python exercise3_linear_regression.py
python exercise4_classification_models.py
```

## 8. What to Explain to Faculty

This project uses a campus electricity dataset to demonstrate data loading, data analysis, regression, and classification.

The full dataset is very large, so a smaller weekly dataset is created for experiments. This makes the programs easier to run while still keeping useful patterns from the original data.

Linear Regression is used to predict electricity consumption. Bayesian Logistic Regression and SVM are used to classify whether a reading is normal or anomalous.
