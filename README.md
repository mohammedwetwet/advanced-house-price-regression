# advanced-house-price-regression
An end-to-end house price regression system using robust preprocessing, K-Fold Cross-Validation, RandomizedSearchCV, and Random Forest optimization.

# 🏠 Advanced House Price Regression

An end-to-end machine learning regression system for predicting house sale prices using the Kaggle House Prices dataset.

This project focuses on building a reliable and reproducible regression workflow, including data preprocessing, model comparison, K-Fold Cross-Validation, hyperparameter optimization with RandomizedSearchCV, and detailed model evaluation.

---

## 📌 Project Overview

The goal of this project is to predict the `SalePrice` of residential properties based on a large set of numerical and categorical features.

Rather than relying on a single train/test split, the project follows a structured machine learning pipeline designed to reduce data leakage and provide more reliable model evaluation.

### Main Workflow

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature / Target Separation
     │
     ▼
Train / Validation Split
     │
     ▼
Data Preprocessing Pipeline
     │
     ├── Numerical → Median Imputation
     │
     └── Categorical → Most-Frequent Imputation
                         + One-Hot Encoding
     │
     ▼
Baseline Model
Linear Regression
     │
     ▼
Flexible Model
Random Forest
     │
     ▼
5-Fold Cross-Validation
     │
     ▼
RandomizedSearchCV
     │
     ▼
Tuned Random Forest
     │
     ▼
Model Comparison
     │
     ▼
Final Evaluation
     │
     ▼
Kaggle Submission


---

🎯 Objectives

The project was designed to:

Explore the target variable and dataset structure.

Separate features and target variables.

Handle missing numerical and categorical values.

Encode categorical variables using One-Hot Encoding.

Build a preprocessing pipeline using Pipeline and ColumnTransformer.

Establish a Linear Regression baseline.

Build a more flexible Random Forest regression model.

Apply 5-Fold Cross-Validation.

Evaluate models using MAE, RMSE, and R².

Perform hyperparameter optimization using RandomizedSearchCV.

Search across a reasonably large hyperparameter space.

Compare baseline, Random Forest, and tuned Random Forest models.

Analyze predictions and residuals visually.

Generate predictions for the Kaggle competition.



---

🧠 Machine Learning Approach

1. Exploratory Data Analysis

The dataset was inspected to understand:

Dataset dimensions

Numerical and categorical features

Missing values

Duplicate records

Target statistics

Target distribution

Target skewness


The target variable is:

SalePrice


---

2. Data Preprocessing

A leakage-aware preprocessing pipeline was created using ColumnTransformer and Pipeline.

Numerical Features

Missing numerical values are handled using:

SimpleImputer(strategy="median")

Categorical Features

Missing categorical values are handled using:

SimpleImputer(strategy="most_frequent")

Categorical variables are then transformed using:

OneHotEncoder(handle_unknown="ignore")

This preprocessing is integrated directly into the model pipeline to prevent information leakage during cross-validation.


---

🌲 Models

Baseline — Linear Regression

Linear Regression was used as a simple baseline model.

It provides a reference point for evaluating whether a more flexible model can capture additional relationships in the data.

Random Forest Regressor

A Random Forest Regressor was used as a more flexible nonlinear model.

The model combines multiple decision trees and aggregates their predictions to improve generalization.


---

🔬 Cross-Validation

The project uses 5-Fold Cross-Validation:

KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

Cross-validation provides a more reliable estimate of model performance across different subsets of the training data.

The following metrics are evaluated:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R²

Mean CV Score

Standard Deviation of CV Score



---

⚙️ Hyperparameter Optimization

The Random Forest model was optimized using:

RandomizedSearchCV

Instead of testing only a few manually selected parameters, the search explores a broader hyperparameter space.

The search includes:

n_estimators

max_depth

min_samples_split

min_samples_leaf

max_features

bootstrap


Example search space:

param_distributions = {
    "model__n_estimators": randint(200, 800),
    "model__max_depth": [None, 10, 15, 20, 25, 30, 40],
    "model__min_samples_split": randint(2, 15),
    "model__min_samples_leaf": randint(1, 8),
    "model__max_features": [
        "sqrt",
        "log2",
        0.5,
        0.7,
        1.0
    ],
    "model__bootstrap": [True, False]
}

The search was performed using 5-Fold Cross-Validation.


---

📊 Model Evaluation

The models are compared using:

MAE

Measures the average absolute prediction error.

Lower is better.

RMSE

Penalizes larger prediction errors more strongly.

Lower is better.

R²

Measures the proportion of target variance explained by the model.

Higher is better.


---

📈 Visual Analysis

The project includes several model diagnostics and visualizations.

Actual vs Predicted

Compares real house prices against model predictions.

A strong model should produce predictions close to the diagonal reference line.

Residual Analysis

Residuals are calculated as:

Residual = Actual Price - Predicted Price

Residual visualization helps identify systematic prediction errors and potential model limitations.

Prediction Error Distribution

The distribution of prediction errors is also analyzed.

Feature Importance

Random Forest feature importance is extracted to identify the features contributing most to the model's predictions.


---

🏆 Kaggle Result

The final model was submitted to the Kaggle:

House Prices — Advanced Regression Techniques

Result

Metric	Result

Kaggle Score	0.15335
Leaderboard Rank	2687 / 3438
Model	Tuned Random Forest
Hyperparameter Search	RandomizedSearchCV
Cross-Validation	5-Fold K-Fold


The Kaggle submission was generated directly from the final trained model.


---

🛡️ Data Leakage Prevention

A key design decision in this project was keeping preprocessing inside the machine learning pipeline.

Instead of preprocessing the entire dataset before cross-validation, transformations are fitted within each training fold.

This helps prevent information from validation folds from influencing the training process.

The validation set was kept separate from hyperparameter optimization and used for final model evaluation.


---

🧰 Technologies Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

SciPy

Kaggle


Key Scikit-learn Components

Pipeline

ColumnTransformer

SimpleImputer

OneHotEncoder

LinearRegression

RandomForestRegressor

KFold

cross_validate

RandomizedSearchCV



---

📁 Project Structure

advanced-house-price-regression/
│
├── advanced-house-price-regression.ipynb
├── README.md
└── submission.csv


---

🚀 How to Run

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/advanced-house-price-regression.git

2. Open the notebook

Open:

advanced-house-price-regression.ipynb

3. Dataset

The project uses the Kaggle House Prices dataset:

train.csv
test.csv

The notebook was originally designed to run in the Kaggle environment.


---

💡 Key Takeaways

This project demonstrates a complete regression workflow rather than simply training a single model.

The main lessons include:

Proper handling of mixed numerical and categorical data.

Importance of preprocessing pipelines.

Avoiding data leakage.

Using Cross-Validation for more reliable evaluation.

Understanding the difference between model training and hyperparameter tuning.

Using RandomizedSearchCV for larger hyperparameter search spaces.

Comparing simple and flexible regression models.

Evaluating models using multiple regression metrics.

Using residual analysis to understand prediction errors.



---

🔮 Future Improvements

Possible improvements include:

Log-transforming the target variable to better align with Kaggle's RMSLE evaluation metric.

Advanced feature engineering.

Gradient Boosting models.

XGBoost / LightGBM / CatBoost.

Ensemble and stacking methods.

More extensive hyperparameter optimization.

Improved handling of skewed numerical features.



---

👤 Author

Mohammed

Machine Learning & Artificial Intelligence Enthusiast


---

⭐ If you found this project useful, feel free to explore the notebook and the methodology used.
