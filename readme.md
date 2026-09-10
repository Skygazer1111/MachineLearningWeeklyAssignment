# Machine Learning Weekly Assignment - Week 1

This repository is part of my **Machine Learning course progress project**. It contains Week 1 exercises focused on importing, exploring, and analyzing a campus electricity dataset using Python and pandas.

## Project Overview

The goal of this week is to work with real-world energy consumption data from a campus environment and build a foundation for later machine learning tasks such as prediction, anomaly detection, and usage optimization.

**Dataset:** Campus Electricity ML Dataset  
**Topic:** Electricity usage, cost, occupancy, and environmental factors across campus buildings

## Files

| File | Description |
|------|-------------|
| `exercise1_load_view.py` | Imports, loads, and displays an overview of the dataset |
| `exercise2_summary_statistics.py` | Generates summary statistics and meaningful comparisons from the dataset |
| `campus_electricity_ml_dataset.csv` | Dataset file (kept locally; not uploaded to GitHub due to file size limits) |

## Week 1 Tasks

### Exercise 1: Import, Load, and View Dataset
- Load the CSV file using pandas
- Display dataset shape, column types, head, tail, sample rows, and info

### Exercise 2: Summary and Statistics
- Show electricity usage and cost statistics
- Compare usage across buildings, peak/off-peak hours, weekdays/weekends, months, and time slots
- Analyze relationships between temperature, AC usage, and power consumption

## Requirements

- Python 3
- pandas
- numpy

Install dependencies:

```bash
pip install pandas numpy
```

## How to Run

Place `campus_electricity_ml_dataset.csv` in the same folder as the scripts, then run:

```bash
python exercise1_load_view.py
python exercise2_summary_statistics.py
```

## Notes

- This project is maintained as part of ongoing weekly ML coursework.
- The dataset is excluded from Git tracking because it exceeds GitHub's 100 MB file size limit.
