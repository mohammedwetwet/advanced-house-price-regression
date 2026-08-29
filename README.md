# 🏠 EstateIQ — Advanced House Price Prediction

An end-to-end machine learning system for predicting house prices using the Kaggle House Prices dataset.

The project combines a complete machine learning workflow with an interactive web application called **EstateIQ**, allowing users to enter property information and receive an estimated house value.

---

## 🚀 Project Overview

**EstateIQ** is a machine learning-powered house price prediction application built around a trained Random Forest regression model.

The project covers the complete workflow:

```text
Kaggle House Prices Dataset
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Data Cleaning & Preprocessing
          │
          ▼
Feature Engineering
          │
          ▼
Train / Validation Split
          │
          ▼
Random Forest Regression
          │
          ▼
5-Fold Cross-Validation
          │
          ▼
RandomizedSearchCV
          │
          ▼
Optimized Model
          │
          ▼
Saved Model
(model.joblib)
          │
          ▼
EstateIQ Web Application
          │
          ▼
Property Value Estimation


---

🎯 Main Features

🤖 Machine Learning

Complete house price regression workflow

Numerical and categorical feature preprocessing

Missing value handling

One-Hot Encoding

Scikit-learn Pipelines

ColumnTransformer

Random Forest Regression

5-Fold Cross-Validation

RandomizedSearchCV

Multiple regression evaluation metrics

Model persistence using Joblib


🌐 EstateIQ Application

The trained model is integrated into an interactive web application.

Users can:

Enter property information

Submit the property details

Get an estimated property value

Interact with a simple and user-friendly interface

Learn more about the EstateIQ system


The application uses the saved trained model rather than retraining the model every time the application runs.


---

🧠 Machine Learning Approach

1. Exploratory Data Analysis

The dataset was analyzed to understand:

Dataset dimensions

Numerical features

Categorical features

Missing values

Duplicate records

Target statistics

Target distribution

Feature relationships


The target variable is:

SalePrice


---

2. Data Preprocessing

A leakage-aware preprocessing pipeline was implemented using:

Pipeline

ColumnTransformer

SimpleImputer

OneHotEncoder


Numerical Features

Missing numerical values are handled using median imputation:

SimpleImputer(strategy="median")

Categorical Features

Missing categorical values are handled using:

SimpleImputer(strategy="most_frequent")

Categorical features are then transformed using:

OneHotEncoder(handle_unknown="ignore")

Keeping preprocessing inside the machine learning pipeline helps prevent data leakage during cross-validation and model training.


---

🌲 Model

Random Forest Regressor

The main prediction model is a:

RandomForestRegressor

Random Forest was selected because it can capture nonlinear relationships between property features and house prices while working effectively with a large number of input features.

The model was optimized using:

RandomizedSearchCV


---

🔬 Cross-Validation

The project uses 5-Fold Cross-Validation:

KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

Cross-validation provides a more reliable estimate of model performance by evaluating the model across multiple training and validation folds.

The following metrics were considered:

MAE

RMSE

R²

Mean CV Score

Standard Deviation of CV Score



---

⚙️ Hyperparameter Optimization

The Random Forest model was optimized using:

RandomizedSearchCV

The search explored several important Random Forest parameters, including:

n_estimators

max_depth

min_samples_split

min_samples_leaf

max_features

bootstrap


This allowed the model to search across a broader parameter space instead of relying only on manually selected values.


---

📊 Model Evaluation

The model was evaluated using several regression metrics.

MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

Lower is better.

RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

Lower is better.

R² — R-Squared

Measures how much of the variance in the target variable is explained by the model.

Higher is better.


---

🏆 Kaggle Result

The model was submitted to the Kaggle:

House Prices — Advanced Regression Techniques

Metric	Result

Kaggle Score	0.15335
Leaderboard Rank	2687 / 3438
Model	Tuned Random Forest
Hyperparameter Search	RandomizedSearchCV
Cross-Validation	5-Fold K-Fold


The Kaggle submission was generated using predictions from the final trained model.


---

🛡️ Data Leakage Prevention

One of the main design decisions in this project was keeping preprocessing inside the machine learning pipeline.

Instead of preprocessing the complete dataset before cross-validation, preprocessing transformations are fitted as part of each training process.

This helps prevent information from validation data from leaking into the training process.

The validation set was kept separate from hyperparameter optimization and used for final model evaluation.


---

🌐 EstateIQ Application

The final trained model was integrated into the EstateIQ application.

The application loads the trained model and the required metadata from saved Joblib files.

Application Components

app.py
   │
   ├── Loads model.joblib
   │
   ├── Loads metadata.joblib
   │
   ├── Receives property information
   │
   ├── Prepares user input
   │
   └── Generates predicted property value

This makes it possible to use the trained machine learning model through an interactive application without running the complete training process again.


---

📁 Project Structure

advanced-house-price-regression/
│
├── app.py
├── advanced-house-price-regression.ipynb
├── model.joblib
├── metadata.joblib
├── requirements.txt
├── submission.csv
└── README.md

File Description

File	Description

app.py	EstateIQ application
advanced-house-price-regression.ipynb	Machine learning development and experimentation
model.joblib	Trained machine learning model
metadata.joblib	Saved information required by the application for processing inputs
requirements.txt	Python dependencies
submission.csv	Kaggle submission predictions
README.md	Project documentation



---

🛠️ Technologies Used

Programming Language

Python


Data Science

NumPy

Pandas


Machine Learning

Scikit-learn

SciPy

Joblib


Visualization

Matplotlib

Seaborn


Application

Streamlit


Dataset

Kaggle House Prices Dataset



---

🚀 How to Run

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/advanced-house-price-regression.git

cd advanced-house-price-regression


---

2. Install Dependencies

pip install -r requirements.txt


---

3. Run EstateIQ

streamlit run app.py

The application will open in your browser.


---

📊 Dataset

This project uses the Kaggle:

House Prices — Advanced Regression Techniques

The original dataset contains residential property information and the target variable:

SalePrice

The model was developed using the Kaggle training data and generated predictions for the competition test data.


---

💡 Key Takeaways

This project demonstrates a complete machine learning workflow, from raw data to an interactive prediction application.

The main concepts demonstrated include:

Data exploration

Mixed numerical and categorical data preprocessing

Missing value handling

One-Hot Encoding

Machine learning pipelines

Data leakage prevention

Random Forest regression

Cross-validation

Hyperparameter optimization

Model evaluation

Model persistence

Building an interactive ML application

Connecting a trained ML model to a user-facing interface



---

🔮 Future Improvements

Possible future improvements include:

Log-transforming the target variable to better align with Kaggle's RMSLE metric

Advanced feature engineering

Gradient Boosting models

XGBoost

LightGBM

CatBoost

Ensemble learning

Stacking multiple regression models

More extensive hyperparameter optimization

Improved handling of skewed numerical features

Model explainability

Additional EstateIQ features



---

👤 Author

Mohammed Adel Yousef Wetwet

Machine Learning & Artificial Intelligence Enthusiast

Interested in:

Artificial Intelligence

Machine Learning

Computer Vision

Natural Language Processing

AI Research



---

⭐ If you found this project useful, feel free to explore the repository and the methodology behind EstateIQ.
