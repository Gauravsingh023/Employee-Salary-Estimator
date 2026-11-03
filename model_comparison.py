import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "employee_salary_regression.csv"
MODEL_FILE = "salary_model.pkl"
METRICS_FILE = "model_metrics.csv"

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

# 2. Features and target
X = df.drop(columns=["employee_id", "annual_salary_usd"])
y = df["annual_salary_usd"]

categorical_features = ["education_level", "job_role"]
numeric_features = [
    "age",
    "years_experience",
    "city_tier",
    "performance_score",
    "num_skills",
    "remote_work",
]

# 3. Preprocessing
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Candidate models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, random_state=42
    ),
}

results = []
trained_pipelines = {}

# 6. Train and evaluate
for name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )

    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4),
        }
    )
    trained_pipelines[name] = pipeline

results_df = pd.DataFrame(results).sort_values(
    "R2 Score", ascending=False
).reset_index(drop=True)

# 7. Save comparison results
results_df.to_csv(METRICS_FILE, index=False)

# 8. Save the best model
best_model_name = results_df.loc[0, "Model"]
best_pipeline = trained_pipelines[best_model_name]
joblib.dump(best_pipeline, MODEL_FILE)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))
print("\n========== BEST MODEL ==========")
print(f"Model: {best_model_name}")
print(f"MAE: {results_df.loc[0, 'MAE']}")
print(f"RMSE: {results_df.loc[0, 'RMSE']}")
print(f"R2 Score: {results_df.loc[0, 'R2 Score']}")
print(f"\nSaved: {MODEL_FILE}")
print(f"Saved: {METRICS_FILE}")
