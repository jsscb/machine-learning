# Customer Churn Prediction

This project focuses on building machine learning models to predict customer churn using structured tabular data. The workflow includes Exploratory Data Analysis (EDA), preprocessing, and training classification models with hyperparameter tuning.

## 📊 Dataset
- Total Records: 41,259 rows × 14 columns
- Numerical Features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`
- Categorical Features: `CustomerId`, `Surname`, `Geography`, `Gender`, `HasCrCard`, `IsActiveMember`, `churn`
- Target Variable: `churn` (Binary classification)
- Note: Missing values were detected in `CreditScore`

## ⚙️ Preprocessing Steps
1. **Missing Value Handling** – Filled missing values in `CreditScore`
2. **Encoding** – Categorical variables were encoded using appropriate methods
3. **Scaling** – Numerical features were scaled for model robustness
4. **Train-Test Split** – Dataset was split for model training and evaluation

## 🧠 Models Used
- **Random Forest (RF)**
  - Trained with both original and scaled data
  - Hyperparameter tuning using `GridSearchCV`
- **XGBoost (XGB)**
  - Trained with both original and scaled data
  - Hyperparameter tuning for performance optimization
