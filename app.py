import os
import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "salary_model.pkl"
DATA_FILE = "employee_salary_regression.csv"
METRICS_FILE = "model_metrics.csv"

st.set_page_config(
    page_title="Employee Salary & Compensation Estimator",
    page_icon="💰",
    layout="wide",
)

# ------------------------------------------------------------
# Load project assets
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_FILE):
        return pd.read_csv(METRICS_FILE)
    return pd.DataFrame()

try:
    model = load_model()
    df = load_data()
    metrics_df = load_metrics()
except Exception as e:
    st.error(f"❌ Project asset could not be loaded: {e}")
    st.stop()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("💰 Employee Salary & Compensation Estimator")
st.markdown(
    "**Predict annual employee compensation using Machine Learning**"
)
st.write(
    "Use the employee profile controls to estimate salary and compare "
    "the prediction with the observed market benchmark for the selected role."
)

st.divider()

# ------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------
st.sidebar.header("👤 Employee Information")

age = st.sidebar.number_input(
    "Age", min_value=18, max_value=70, value=25, step=1
)

years_experience = st.sidebar.slider(
    "Years of Experience", min_value=0, max_value=40, value=2, step=1
)

education_options = sorted(df["education_level"].dropna().unique().tolist())
education_level = st.sidebar.selectbox(
    "Education Level", education_options
)

job_roles = sorted(df["job_role"].dropna().unique().tolist())
job_role = st.sidebar.selectbox("Job Role", job_roles)

city_tiers = sorted(df["city_tier"].dropna().unique().tolist())
city_tier = st.sidebar.selectbox("City Tier / Geography", city_tiers)

performance_score = st.sidebar.slider(
    "Performance Score", min_value=0.0, max_value=5.0,
    value=3.0, step=0.1
)

num_skills = st.sidebar.slider(
    "Number of Skills", min_value=1, max_value=15, value=5, step=1
)

remote_work = st.sidebar.selectbox("Remote Work", ["No", "Yes"])
remote_value = 1 if remote_work == "Yes" else 0

predict_clicked = st.button(
    "💰 Predict Salary", use_container_width=True, type="primary"
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if predict_clicked:
    input_data = pd.DataFrame({
        "age": [age],
        "years_experience": [years_experience],
        "education_level": [education_level],
        "job_role": [job_role],
        "city_tier": [city_tier],
        "performance_score": [performance_score],
        "num_skills": [num_skills],
        "remote_work": [remote_value],
    })

    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.dataframe(input_data)
        st.stop()

    role_data = df[df["job_role"] == job_role]
    market_average = float(role_data["annual_salary_usd"].mean())
    market_min = float(role_data["annual_salary_usd"].min())
    market_max = float(role_data["annual_salary_usd"].max())

    difference = prediction - market_average
    percentage_difference = (
        (difference / market_average) * 100
        if market_average else 0
    )

    # Salary prediction cards
    st.subheader("🎯 Salary Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Estimated Annual Salary", f"${prediction:,.0f}")

    with c2:
        st.metric("Market Average", f"${market_average:,.0f}")

    with c3:
        st.metric(
            "Difference",
            f"${difference:,.0f}",
            f"{percentage_difference:+.1f}%"
        )

    st.divider()

    # Benchmark
    st.subheader("📊 Salary Benchmark")
    if prediction > market_average:
        st.success(
            "🟢 **Above Market Average**\n\n"
            "The predicted salary is higher than the average salary "
            "observed for this job role."
        )
    elif prediction < market_average:
        st.warning(
            "🟡 **Below Market Average**\n\n"
            "The predicted salary is lower than the average salary "
            "observed for this job role."
        )
    else:
        st.info(
            "🔵 **At Market Average**\n\n"
            "The predicted salary is approximately equal to the "
            "observed role average."
        )

    # Market range
    st.subheader("💵 Market Salary Range")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Minimum Observed Salary", f"${market_min:,.0f}")
    with r2:
        st.metric("Maximum Observed Salary", f"${market_max:,.0f}")

    # Salary position
    st.subheader("📈 Salary Position")
    if market_max > market_min:
        position = ((prediction - market_min) /
                    (market_max - market_min)) * 100
        position = max(0.0, min(100.0, position))
        st.progress(
            int(position),
            text=f"Predicted salary position: {position:.1f}%"
        )

    # Profile
    st.divider()
    st.subheader("👤 Selected Employee Profile")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.write(f"**Age:** {age}")
        st.write(f"**Experience:** {years_experience} years")
        st.write(f"**Education:** {education_level}")

    with p2:
        st.write(f"**Job Role:** {job_role}")
        st.write(f"**City Tier:** {city_tier}")
        st.write(f"**Remote Work:** {remote_work}")

    with p3:
        st.write(f"**Performance:** {performance_score}/5")
        st.write(f"**Number of Skills:** {num_skills}")

# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------
st.divider()
st.subheader("🤖 Machine Learning Model")

if not metrics_df.empty:
    best = metrics_df.iloc[0]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Best Model", str(best["Model"]))
    with m2:
        st.metric("R² Score", f"{float(best['R2 Score']):.4f}")
    with m3:
        st.metric("MAE", f"${float(best['MAE']):,.2f}")

    st.caption(
        "The final model is selected using the highest R² score on the "
        "held-out test set."
    )

    with st.expander("📋 Compare All Models"):
        display_metrics = metrics_df.copy()
        display_metrics["MAE"] = display_metrics["MAE"].map(
            lambda x: f"${float(x):,.2f}"
        )
        display_metrics["RMSE"] = display_metrics["RMSE"].map(
            lambda x: f"${float(x):,.2f}"
        )
        display_metrics["R2 Score"] = display_metrics["R2 Score"].map(
            lambda x: f"{float(x):.4f}"
        )
        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Run model_comparison.py to generate model_metrics.csv.")

# ------------------------------------------------------------
# Dataset and project information
# ------------------------------------------------------------
with st.expander("📊 Dataset Overview"):
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Records", f"{len(df):,}")
    with d2:
        st.metric("Features", f"{df.shape[1] - 2}")
    with d3:
        st.metric("Missing Values", f"{int(df.isnull().sum().sum())}")

    st.write("**Target:** annual_salary_usd")
    st.write(
        "**Input features:** age, years of experience, education level, "
        "job role, city tier, performance score, number of skills, "
        "and remote work."
    )

with st.expander("ℹ️ About This Project"):
    st.markdown(
        """
### Objective
Build a Machine Learning application that estimates annual employee
compensation and provides a salary benchmark.

### Methodology
1. Load and validate the employee salary dataset.
2. Perform exploratory data analysis.
3. Encode categorical variables using One-Hot Encoding.
4. Scale numeric variables using StandardScaler.
5. Split data into training and testing sets.
6. Compare Linear Regression, Random Forest and Gradient Boosting.
7. Select the model with the highest R² score.
8. Deploy the final model through Streamlit.
9. Compare predicted salary with the observed role-level market range.

### Models Compared
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
"""
    )

st.caption(
    "Employee Salary & Compensation Estimator | "
    "Machine Learning + Streamlit"
)
