# Credit Risk Classification Model

A binary classification project to predict customer loan default probability using machine learning techniques.

🔗 **Live Demo**: [Credit Risk Model on Streamlit](https://credit-risk-model-end-to-end.streamlit.app/)

## 📋 Project Overview

This project builds a predictive model to assess credit risk by classifying whether a customer is likely to default on their loan. The model helps financial institutions make informed lending decisions by analyzing customer demographics, loan characteristics, and credit bureau data.

## 📊 Dataset

The project uses three merged datasets containing comprehensive customer and loan information:

- **Total Records**: 50,000 rows
- **Total Features**: 33 columns
- **Data Sources**: Customer data, Loan data, and Credit Bureau data

### Target Variable
- **default**: Binary classification (0 = No Default, 1 = Default)
  - Class 0 (No Default): 45,703 samples (91.4%)
  - Class 1 (Default): 4,297 samples (8.6%)
  - **Note**: Significant class imbalance handled using SMOTE

### Key Features
- Customer demographics (age, gender, marital status, employment, income, dependents)
- Residence information (type, years at address, location)
- Loan details (purpose, type, amount, tenure, disbursement info)
- Credit bureau metrics (open/closed accounts, delinquency, credit utilization)

## 🔧 Data Preprocessing

### Data Cleaning
- **Missing Values**: 47 missing values in `residence_type` imputed with mode (Owned)
- **Duplicates**: No duplicates found
- **Outliers**: Removed 5 records where processing fee ratio exceeded 3% of loan amount (business rule)
- **Typos**: Fixed categorical inconsistencies (e.g., "Personnal" → "Personal")

### Feature Engineering
Three new features created to enhance model performance:

1. **loan_to_income**: Ratio of loan amount to customer income
2. **delinquency_ratio**: Proportion of delinquent accounts
3. **avg_dpd_per_delinquency**: Average days past due per delinquent account

### Feature Selection
Selected 10 features using VIF (Variance Inflation Factor), Information Value (IV), and Weight of Evidence (WoE):

- `age`
- `residence_type`
- `loan_purpose`
- `loan_type`
- `loan_tenure_months`
- `number_of_open_accounts`
- `credit_utilization_ratio`
- `loan_to_income`
- `delinquency_ratio`
- `avg_dpd_per_delinquency`

## 🔍 Exploratory Data Analysis Insights

Key findings from EDA:

- **Age**: Younger customers show higher default rates
- **Loan Tenure**: Longer repayment periods (50-60 months) correlate with higher default risk
- **Credit History**: Extended borrowing periods (48+ months) increase default probability
- **Payment Behavior**: Higher delinquency amounts and longer DPD periods predict default
- **Credit Utilization**: High utilization rates immediately after disbursement indicate increased risk

## 🤖 Model Development

### Models Evaluated
Four classification algorithms were trained and compared:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **LightGBM Classifier**

### Handling Class Imbalance
- **Technique**: SMOTE (Synthetic Minority Over-sampling Technique)
- Balanced the training set to improve minority class (default) detection

### Hyperparameter Optimization
- **Framework**: Optuna for automated hyperparameter tuning
- Conducted optimization studies to find best model configurations

## 📈 Model Performance

### Logistic Regression (Selected Model)
```
              precision    recall    f1-score    support
           0       0.99      0.93      0.96      11423
           1       0.56      0.94      0.70       1074
    accuracy                           0.93      12497
```

### Random Forest Classifier
```
              precision    recall    f1-score    support
           0       0.99      0.96      0.98      11423
           1       0.70      0.86      0.77       1074
    accuracy                           0.96      12497
```

### XGBoost Classifier
```
              precision    recall    f1-score    support
           0       0.98      0.97      0.98      11423
           1       0.74      0.83      0.78       1074
    accuracy                           0.96      12497
```

### LightGBM Classifier
```
              precision    recall    f1-score    support
           0       0.99      0.96      0.98      11423
           1       0.70      0.88      0.78       1074
    accuracy                           0.96      12497
```

## 🏆 Final Model Selection

**Selected Model: Logistic Regression**

**Rationale**: Despite similar performance across all models, Logistic Regression was chosen as the final model due to its superior **explainability**. In credit risk assessment, regulatory compliance and stakeholder transparency are critical, making model interpretability essential for:

- Clear coefficient interpretation
- Regulatory audit requirements
- Stakeholder communication
- Feature importance transparency
- Business rule validation

### Model Evaluation Metrics
- **ROC-AUC Score**: Evaluated discrimination capability
- **Rank Ordering**: Assessed predictive power across score segments
- **Kolmogorov-Smirnov (KS) Statistic**: Measured separation between classes
- **Gini Coefficient**: Evaluated model inequality measures

### Feature Importance
Top predictive features from the logistic regression model:

1. `loan_to_income` (highest importance)
2. `credit_utilization_ratio`
3. `delinquency_ratio`
4. `avg_dpd_per_delinquency`
5. `residence_type_Rented`

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`, `lightgbm`
  - Imbalance Handling: `imbalanced-learn` (SMOTE)
  - Hyperparameter Tuning: `optuna`
  - Deployment: `streamlit`

## 🚀 Deployment

The model is deployed as an interactive web application using Streamlit, allowing users to:
- Input customer and loan details
- Receive real-time default probability predictions
- View feature importance and model explanations

Access the live application: [https://credit-risk-model-end-to-end.streamlit.app/](https://credit-risk-model-end-to-end.streamlit.app/)



## 💡 Key Takeaways

1. **Class imbalance** significantly impacts model performance; SMOTE effectively addresses this issue
2. **Feature engineering** (loan-to-income, delinquency metrics) substantially improves predictive power
3. **Model explainability** is crucial in financial applications for regulatory compliance
4. **Younger customers** and those with **longer loan tenures** present higher default risk
5. **Credit utilization patterns** and **payment history** are strong default predictors

## 🔮 Future Improvements

- Implement ensemble methods combining multiple models
- Add real-time data pipeline integration
- Incorporate additional external data sources (macroeconomic indicators)
- Develop model monitoring dashboard for production deployment
- Implement A/B testing framework for model comparison
- Add explainable AI features (SHAP values, LIME)

## 📝 License

This project is available for educational and reference purposes.

---

**Note**: This model is designed for educational purposes and should undergo rigorous validation, regulatory review, and bias testing before production deployment in financial decision-making systems.
