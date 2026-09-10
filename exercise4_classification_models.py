# Exercise 4: Bayesian Logistic Regression and SVM for Classification
# Dataset: Reduced Campus Electricity Weekly Experiment Dataset

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


DATASET_FILE = "campus_electricity_weekly_experiment.csv"
target_column = "injected_anomaly"

if not Path(DATASET_FILE).exists():
    raise SystemExit(
        f"{DATASET_FILE} not found. Run prepare_weekly_experiment_dataset.py first."
    )

# Load the reduced weekly dataset
df = pd.read_csv(DATASET_FILE)

# Features used for classification
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
    "power_kw",
    "energy_kwh_15min",
    "tariff_inr_per_kwh",
    "energy_cost_inr",
]

categorical_features = [
    "building_id",
    "building_type",
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df[target_column]

# Stratify keeps the rare anomaly class represented in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Preprocessing:
# - Fill missing values
# - Scale numeric columns for Logistic Regression and SVM
# - One-hot encode categorical columns
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
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

# Bayesian Logistic Regression:
# Scikit-learn does not provide full posterior Bayesian Logistic Regression.
# L2-regularized Logistic Regression is a common MAP approximation using a
# Gaussian prior on coefficients, which is suitable for this course experiment.
bayesian_logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

# Support Vector Machine classifier
svm_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                random_state=42,
            ),
        ),
    ]
)


def evaluate_model(model_name, model, show_probabilities=False):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n" + "=" * 65)
    print(f"  {model_name}")
    print("=" * 65)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions)
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    )
    print("Confusion Matrix:")
    print(matrix_df.to_string())

    sample_output = pd.DataFrame(
        {
            "Actual_Class": y_test.head(10).values,
            "Predicted_Class": predictions[:10],
        }
    )

    if show_probabilities:
        probabilities = model.predict_proba(X_test)[:10, 1]
        sample_output["Anomaly_Probability"] = probabilities

    print("\nSample Predictions:")
    print(sample_output.round(3).to_string(index=False))

    return predictions


print("=" * 65)
print("  BAYESIAN LOGISTIC REGRESSION AND SVM CLASSIFICATION")
print("=" * 65)

print("\n>> CLASSIFICATION OBJECTIVE")
print("  Classify whether an electricity reading is normal or anomalous.")

print("\n>> DATASET USED")
print(f"  Dataset File        : {DATASET_FILE}")
print(f"  Total Records       : {len(df):,}")
print(f"  Training Records    : {len(X_train):,}")
print(f"  Testing Records     : {len(X_test):,}")
print(f"  Number of Features  : {len(feature_columns)}")

print("\n>> TARGET CLASS DISTRIBUTION")
class_counts = y.value_counts().sort_index()
print(f"  Normal readings     : {class_counts.get(0, 0):,}")
print(f"  Anomaly readings    : {class_counts.get(1, 0):,}")
print(f"  Anomaly percentage  : {(class_counts.get(1, 0) / len(df)) * 100:.2f}%")

print("\n>> FEATURES USED")
print("  Electricity Values  : power, energy consumed, cost, tariff")
print("  Usage Values        : AC usage, lighting usage, plug load")
print("  Context Values      : occupancy, weather, building, room, time, peak period")

evaluate_model(
    "BAYESIAN LOGISTIC REGRESSION (MAP with L2 Gaussian Prior)",
    bayesian_logistic_model,
    show_probabilities=True,
)

evaluate_model("SUPPORT VECTOR MACHINE (Linear SVM)", svm_model)

print("\n" + "=" * 65)
print("  CONCLUSION")
print("=" * 65)
print(
    "  Both models can be used for binary classification. Bayesian Logistic "
    "Regression gives class probabilities, while SVM focuses on finding a "
    "strong separating boundary between normal and anomalous readings."
)
