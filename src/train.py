# src/train.py

"""
train.py — Final training script (recruitment-ready)

Trains multiple ML models, compares them using MAE, RMSE, and R²,
selects the best one (by RMSE), and saves the full pipeline
(preprocessing + model) to: models/best_model.pkl
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib


# ─────────────────────────────────────────────────────────────
# PATHS (ROBUST TO RUN LOCATION)
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "car_price_prediction_.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    print("\nPreview:")
    print(data.head())

    print("\nInfo:")
    print(data.info())

    # ─────────────────────────────────────────────────────────
    # FEATURES & TARGET
    # ─────────────────────────────────────────────────────────

    numeric_features = ["Mileage", "Engine Size", "Year"]
    categorical_features = ["Fuel Type", "Brand", "Condition"]

    X = data[numeric_features + categorical_features]
    y = data["Price"]

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # ─────────────────────────────────────────────────────────
    # TRAIN / TEST SPLIT
    # ─────────────────────────────────────────────────────────

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ─────────────────────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────────────────────

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ─────────────────────────────────────────────────────────
    # MODELS
    # ─────────────────────────────────────────────────────────

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    # ─────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────

    trained_pipelines = {}
    results = []

    for name, model in models.items():
        print(f"\nTraining model: {name}")

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(X_train, y_train)

        trained_pipelines[name] = clf

        y_pred = clf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)

    print("\nModel comparison (sorted by RMSE):")
    print(results_df.to_string(index=False))

    # ─────────────────────────────────────────────────────────
    # SAVE BEST MODEL
    # ─────────────────────────────────────────────────────────

    MODELS_DIR.mkdir(exist_ok=True)

    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved to: {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
