# Car Price Prediction

This project uses a publicly available dataset containing car attributes and prices. The data was used for educational and portfolio purposes.

An end-to-end Machine Learning project that predicts car prices based on structured vehicle attributes.

The project demonstrates the full ML workflow, including exploratory data analysis, preprocessing, model training and evaluation, model persistence, and deployment via a Streamlit web application.

---

## Overview

This project focuses on building a clean, reproducible Machine Learning pipeline for a regression problem.
Rather than optimizing for maximum accuracy, the goal is to demonstrate best practices in applied ML,
such as consistent preprocessing, fair model comparison, and clear separation between training and inference.

---

## Problem Context

Car price prediction is a common real-world regression problem in applied machine learning.
Prices depend on multiple interacting factors, and real datasets often contain noise
and limited feature information.

This project emphasizes:
- Proper handling of mixed numerical and categorical features
- Robust preprocessing using pipelines
- Objective model comparison using standard regression metrics

---

## Project structure

```text
Car_Price_Prediction/
├── data/
│   └── car_price_prediction_.csv
├── src/
│   ├── eda.py
│   └── train.py
├── models/
│   └── best_model.pkl
├── archive/                # older experiments (optional)
├── app.py
├── requirements.txt
└── README.md
```

## Features

**Numeric:**
- Mileage
- Engine Size
- Year

**Categorical:**
- Fuel Type
- Brand
- Condition

**Target:**
- Price

## Modeling Approach

- Separate preprocessing pipelines for numerical and categorical features
- Missing value imputation, scaling, and one-hot encoding
- Unified training and evaluation pipeline for all models
- Model selection based on RMSE evaluated on a held-out test set

---

## Models

The following models are trained and compared:
- Linear Regression
- Random Forest
- XGBoost
- LightGBM

The best-performing model (based on the lowest RMSE) is selected and saved to:
`models/best_model.pkl`


---

## Results and Limitations

Due to the limited number of features and the simplified nature of the dataset,
the achieved R² scores are modest. This behavior is expected and highlights
the importance of feature richness and data quality in real-world pricing problems.

The primary focus of this project is to demonstrate a complete and reproducible
Machine Learning workflow rather than achieving state-of-the-art predictive performance.

---

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

### 1) Train and save the model
```powershell
python src/train.py
```

### 2) Run the web application
```powershell
streamlit run app.py
```

### 3) Exploratory Data Analysis (optional)
```powershell
python src/eda.py
```

## Notes

- The trained model file (`models/best_model.pkl`) is generated locally after running training.
- The `archive/` folder contains older experimental versions of training scripts.

## Future Improvements

Additional feature engineering (e.g. vehicle age, usage patterns)
Cross-validation and hyperparameter tuning
Integration with a larger and more realistic dataset

---
