import os
import joblib
import pandas as pd

DATA_FILE = "employee_salary_regression.csv"
MODEL_FILE = "salary_model.pkl"

required_columns = {
    "employee_id",
    "age",
    "years_experience",
    "education_level",
    "job_role",
    "city_tier",
    "performance_score",
    "num_skills",
    "remote_work",
    "annual_salary_usd",
}

assert os.path.exists(DATA_FILE), "Dataset not found."
assert os.path.exists(MODEL_FILE), "Model file not found."

df = pd.read_csv(DATA_FILE)

assert required_columns.issubset(df.columns), "Dataset columns are incomplete."
assert len(df) > 0, "Dataset is empty."
assert df.isnull().sum().sum() == 0, "Dataset contains missing values."

model = joblib.load(MODEL_FILE)

sample = pd.DataFrame({
    "age": [25],
    "years_experience": [2],
    "education_level": ["Bachelor"],
    "job_role": ["Data Analyst"],
    "city_tier": [1],
    "performance_score": [3.0],
    "num_skills": [5],
    "remote_work": [1],
})

prediction = float(model.predict(sample)[0])

assert prediction > 0, "Prediction must be positive."

print("All project smoke tests passed.")
print(f"Sample predicted salary: ${prediction:,.2f}")
