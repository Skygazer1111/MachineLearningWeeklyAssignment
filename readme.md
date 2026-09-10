# Machine Learning Weekly Assignment - Week 1

This repository is part of my **Machine Learning course progress project**. It contains Week 1 exercises focused on importing, exploring, and analyzing a campus electricity dataset using Python and pandas.

## Project Overview

The goal of this week is to work with real-world energy consumption data from a campus environment and build a foundation for later machine learning tasks such as prediction, anomaly detection, and usage optimization.

**Dataset:** Campus Electricity ML Dataset  
**Topic:** Electricity usage, cost, occupancy, and environmental factors across campus buildings

## Dataset Explanation

The original dataset is a large campus electricity dataset with more than 1.5 million records. It contains electricity readings taken every 15 minutes for different rooms and buildings on a campus.

Each row represents one electricity reading for a specific room at a specific time. The dataset includes building details, room details, occupancy, weather conditions, appliance usage, electricity consumption, and cost.

For weekly ML experiments, the full dataset has been reduced to a smaller file:

**Reduced Dataset:** `campus_electricity_weekly_experiment.csv`

This reduced dataset keeps:
- 7 days of readings
- 5 campus buildings
- 20 rooms total
- 25 useful columns
- 13,440 records

This size is easier to run on a laptop and is suitable for showing how different machine learning models work.

## Larger-Level Impact

This type of dataset is useful for understanding and predicting electricity demand in large campuses, offices, smart buildings, and cities. By analyzing energy usage patterns, organizations can reduce electricity wastage, plan peak-hour energy usage, lower costs, and improve sustainability.

Machine learning models trained on this kind of data can help predict future electricity consumption, detect abnormal usage, compare building efficiency, and support smart energy management systems.

## Files

| File | Description |
|------|-------------|
| `prepare_weekly_experiment_dataset.py` | Creates a smaller weekly experiment dataset from the full CSV |
| `campus_electricity_weekly_experiment.csv` | Smaller dataset used for weekly ML experiments |
| `exercise1_load_view.py` | Imports, loads, and displays an overview of the dataset |
| `exercise2_summary_statistics.py` | Generates summary statistics and meaningful comparisons from the dataset |
| `exercise3_linear_regression.py` | Implements Linear Regression to predict electricity usage (`power_kw`) |
| `campus_electricity_ml_dataset.csv` | Original full dataset file, kept locally due to GitHub file size limits |

## Week 1 Tasks

### Exercise 1: Import, Load, and View Dataset
- Load the reduced weekly CSV file using pandas
- Display dataset shape, column types, head, tail, sample rows, and info

### Exercise 2: Summary and Statistics
- Show electricity usage and cost statistics
- Compare usage across buildings, peak/off-peak hours, weekdays/weekends, months, and time slots
- Analyze relationships between temperature, AC usage, and power consumption

### Exercise 3: Linear Regression Prediction
- Train a Linear Regression model to predict electricity consumption (`power_kw`)
- Use campus, occupancy, weather, appliance usage, and time-based features
- Evaluate the model using MAE, MSE, RMSE, and R2 Score
- Display sample actual vs predicted electricity usage values

## Requirements

- Python 3
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

Place `campus_electricity_ml_dataset.csv` in the same folder as the scripts, then run:

```bash
python prepare_weekly_experiment_dataset.py
python exercise1_load_view.py
python exercise2_summary_statistics.py
python exercise3_linear_regression.py
```

## Notes

- This project is maintained as part of ongoing weekly ML coursework.
- The dataset is excluded from Git tracking because it exceeds GitHub's 100 MB file size limit.
