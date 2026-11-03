import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "employee_salary_regression.csv"
MODEL_FILE = "salary_model.pkl"

df = pd.read_csv(DATA_FILE)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

target = "annual_salary_usd"
X = df.drop(columns=[target, "employee_id"])
y = df[target]

categorical_features = ["education_level", "job_role"]
numeric_features = [
    "age",
    "years_experience",
    "city_tier",
    "performance_score",
    "num_skills",
    "remote_work",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("numeric", StandardScaler(), numeric_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== LINEAR REGRESSION PERFORMANCE ==========")
print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2  : {r2:.4f}")

joblib.dump(model, MODEL_FILE)
print(f"\nModel saved successfully: {MODEL_FILE}")
