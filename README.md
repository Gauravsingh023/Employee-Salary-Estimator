# Employee Salary & Compensation Estimator

## 1. Project Overview

The Employee Salary & Compensation Estimator is a Machine Learning project designed to estimate an employee's annual salary based on various employee-related attributes.

The project uses Machine Learning regression algorithms and provides an interactive web application developed using Streamlit.

## 2. Project Objectives

- Predict employee annual salary using Machine Learning.
- Analyze employee compensation data.
- Compare different Machine Learning models.
- Provide an interactive salary prediction interface.
- Demonstrate the practical application of Data Science and Machine Learning.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib (if used for model saving/loading)

## 4. Dataset Information

Dataset Name: Employee Compensation & Salary Prediction Dataset

Dataset Source: Kaggle

Dataset URL:  
https://www.kaggle.com/datasets/alitaqishah/employee-compensation-and-salary-prediction-dataset

Dataset Size: 1,000 rows and 10 columns

Target Variable: `annual_salary_usd`

### Dataset Features

- Employee ID
- Age
- Years of Experience
- Education Level
- Job Role
- City Tier
- Performance Score
- Number of Skills
- Remote Work
- Annual Salary (USD)

## 5. Project Structure

Employee_Salary_Project/
│
├── app.py
├── salary_prediction.py
├── model_comparison.py
├── eda.py
├── salary_model.pkl
├── employee_salary_regression.csv
├── requirements.txt
└── README.md


Note: This structure describes the expected project files. Update it if your actual folder contains different files.

## 6. Installation Requirements

Install Python and Visual Studio Code.

Open the project folder in VS Code and create a virtual environment:

```bash
python -m venv .venv
```

### Activate Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 7. How to Run the Project

### Step 1: Open the project directory

```powershell
cd "C:\Users\Gaurav Kumar\OneDrive\Desktop\Employee_Salary_Project"
```

### Step 2: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Run the Streamlit application

```powershell
python -m streamlit run app.py
```

### Step 4: Open the application

Open the following URL in your browser:

http://localhost:8501

## 8. Model Training

The model training script is:

```bash
python salary_prediction.py
```

This script is intended to load the dataset, prepare the data, train the Machine Learning model, evaluate its performance, and save the trained model.

Run this script only when training or updating the model is required.

## 9. Machine Learning Algorithms

The project explores regression algorithms such as:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The models can be evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 10. Application Workflow

1. User enters employee details.
2. The application processes the input data.
3. The trained Machine Learning model predicts the salary.
4. The predicted annual salary is displayed.
5. The application may provide salary benchmarking and comparison information.

## 11. Project Limitations

- Predictions depend on the quality and representativeness of the dataset.
- Predicted salary is an estimate, not a guaranteed actual salary.
- Real-world salary may vary depending on company policies, market conditions, location, and skills.
- Model performance on new data may differ from evaluation results.

## 12. Future Improvements

- Use a larger and more diverse dataset.
- Improve model performance through hyperparameter tuning.
- Add more employee attributes.
- Deploy the application online.
- Improve the user interface and salary benchmarking features.

## 13. Author

Gaurav Kumar

MCA (Artificial Intelligence) Student

Data Science Intern

## 14. Acknowledgement

This project was developed as part of Data Science and Machine Learning learning activities.
