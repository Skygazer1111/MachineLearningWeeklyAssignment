# Exercise 3: Linear Regression for Electricity Consumption Prediction
# Dataset: Campus Electricity ML Dataset

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Load the dataset
DATASET_FILE = "campus_electricity_weekly_experiment.csv"
df = pd.read_csv(DATASET_FILE)

# Target variable: the value we want to predict
target_column = "power_kw"

# Features used for prediction
numeric_features = [
    "floor",
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
]

categorical_features = [
    "building_id",
    "building_type",
]

feature_columns = numeric_features + categorical_features

# Prepare input features (X) and output target (y)
X = df[feature_columns]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Preprocessing:
# - Fill missing numeric values with median
# - Fill missing categorical values with most frequent value
# - Convert categorical columns to numbers using one-hot encoding
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Build the Linear Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("=" * 65)
print("     LINEAR REGRESSION - ELECTRICITY USAGE PREDICTION")
print("=" * 65)

print("\n>> MODEL OBJECTIVE")
print("  Predict campus electricity consumption in kW (power_kw).")

print("\n>> DATASET USED")
print(f"  Dataset File        : {DATASET_FILE}")
print(f"  Total Records       : {len(df):,}")
print(f"  Training Records    : {len(X_train):,}")
print(f"  Testing Records     : {len(X_test):,}")
print(f"  Target Variable     : {target_column}")
print(f"  Number of Features  : {len(feature_columns)}")

print("\n>> FEATURES USED FOR PREDICTION")
print("  Building Details    : building_id, building_type, floor, room area, capacity")
print("  Occupancy Details   : occupancy, occupancy_rate")
print("  Weather Details     : temperature, humidity")
print("  Usage Details       : AC usage, lighting usage, plug load")
print("  Time Details        : hour, day of week, peak period, weekend, holiday, exam period")

print("\n>> MODEL PERFORMANCE")
print(f"  Mean Absolute Error : {mae:.2f} kW")
print(f"  Mean Squared Error  : {mse:.2f}")
print(f"  Root Mean Sq. Error : {rmse:.2f} kW")
print(f"  R2 Score            : {r2:.3f}")

if r2 >= 0.75:
    print("  Result              : Strong prediction performance")
elif r2 >= 0.50:
    print("  Result              : Moderate prediction performance")
else:
    print("  Result              : Weak prediction performance")

print("\n>> SAMPLE ACTUAL vs PREDICTED VALUES")
comparison = pd.DataFrame(
    {
        "Actual_power_kw": y_test.head(10).values,
        "Predicted_power_kw": y_pred[:10],
    }
)
comparison["Difference"] = comparison["Actual_power_kw"] - comparison["Predicted_power_kw"]
print(comparison.round(2).to_string(index=False))

print("\n>> IMPORTANT MODEL COEFFICIENTS")
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_
coef_table = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients,
    }
)
coef_table["Absolute_Value"] = coef_table["Coefficient"].abs()
coef_table = coef_table.sort_values("Absolute_Value", ascending=False).head(10)
print(coef_table[["Feature", "Coefficient"]].round(3).to_string(index=False))

print("\n>> CONCLUSION")
print(
    "  Linear Regression can predict electricity usage using campus, weather, "
    "occupancy, appliance usage, and time-based features."
)
