import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "employee_salary_regression.csv"
CHART_DIR = "charts"

df = pd.read_csv(DATA_FILE)
os.makedirs(CHART_DIR, exist_ok=True)

print("========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe(include="all").transpose())

# 1. Salary distribution
plt.figure(figsize=(8, 5))
plt.hist(df["annual_salary_usd"], bins=30)
plt.title("Annual Salary Distribution")
plt.xlabel("Annual Salary (USD)")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/salary_distribution.png", dpi=150)
plt.close()

# 2. Salary vs experience
plt.figure(figsize=(8, 5))
plt.scatter(df["years_experience"], df["annual_salary_usd"], alpha=0.7)
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Annual Salary (USD)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/salary_vs_experience.png", dpi=150)
plt.close()

# 3. Salary by job role
job_salary = df.groupby("job_role")["annual_salary_usd"].mean().sort_values()
plt.figure(figsize=(9, 5))
plt.barh(job_salary.index, job_salary.values)
plt.title("Average Salary by Job Role")
plt.xlabel("Average Annual Salary (USD)")
plt.ylabel("Job Role")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/salary_by_job_role.png", dpi=150)
plt.close()

# 4. Salary by education
education_salary = (
    df.groupby("education_level")["annual_salary_usd"].mean().sort_values()
)
plt.figure(figsize=(8, 5))
plt.bar(education_salary.index, education_salary.values)
plt.title("Average Salary by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Average Annual Salary (USD)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/salary_by_education.png", dpi=150)
plt.close()

# 5. Salary by city tier
city_salary = (
    df.groupby("city_tier")["annual_salary_usd"].mean().sort_index()
)
plt.figure(figsize=(7, 5))
plt.bar(city_salary.index.astype(str), city_salary.values)
plt.title("Average Salary by City Tier")
plt.xlabel("City Tier")
plt.ylabel("Average Annual Salary (USD)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/salary_by_city_tier.png", dpi=150)
plt.close()

# 6. Salary vs performance
plt.figure(figsize=(8, 5))
plt.scatter(df["performance_score"], df["annual_salary_usd"], alpha=0.7)
plt.title("Salary vs Performance Score")
plt.xlabel("Performance Score")
plt.ylabel("Annual Salary (USD)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/salary_vs_performance.png", dpi=150)
plt.close()

# 7. Salary vs skills
plt.figure(figsize=(8, 5))
plt.scatter(df["num_skills"], df["annual_salary_usd"], alpha=0.7)
plt.title("Salary vs Number of Skills")
plt.xlabel("Number of Skills")
plt.ylabel("Annual Salary (USD)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/salary_vs_skills.png", dpi=150)
plt.close()

# 8. Correlation heatmap
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

print("\n========== CORRELATION WITH SALARY ==========")
print(
    correlation["annual_salary_usd"]
    .sort_values(ascending=False)
    .to_string()
)

plt.figure(figsize=(10, 7))
plt.imshow(correlation, interpolation="nearest")
plt.colorbar()
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right",
)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

print("\nEDA completed successfully.")
print(f"Charts saved in: {CHART_DIR}/")
