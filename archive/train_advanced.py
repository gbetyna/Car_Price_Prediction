"""
model_training_advanced.py

Advanced Feature Engineering + porównanie modeli

Co robi ten skrypt (ROZDZIAŁY 1–8):

1. Wczytuje dane z pliku CSV (ceny samochodów).
2. Dodaje lepsze cechy:
   - Age (wiek auta)
   - Mileage_per_year (przebieg na rok)
   - log_mileage (logarytm przebiegu)
   oraz dalej liczy log_price (cel do trenowania).
3. Przygotowuje macierz cech X i zmienne celu y (Price i log_price).
4. Dzieli dane na zbiory treningowy i testowy.
5. Buduje preprocessing (imputacja, skalowanie, One-Hot Encoding).
6. Definiuje zestaw modeli (Linear Regression, Random Forest, XGBoost, LightGBM).
7. Trenuje modele na log_price, wraca do skali Price i liczy metryki (MAE, RMSE, R²).
8. (Opcjonalnie) wybiera najlepszy model i zapisuje go do pliku .pkl.
"""

# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 0: IMPORTY I KONFIGURACJA PODSTAWOWA
# ─────────────────────────────────────────────────────────────

import numpy as np
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

# Ścieżka do pliku z danymi – dopasuj do swojego projektu.
DATA_PATH = "data/car_price_prediction_.csv"

# Rok odniesienia do obliczania wieku auta.
CURRENT_YEAR = 2024


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 1: WCZYTANIE DANYCH
# ─────────────────────────────────────────────────────────────

data = pd.read_csv(DATA_PATH)

print("🔍 ROZDZIAŁ 1: Podgląd danych (5 pierwszych wierszy):")
print(data.head())

print("\nℹ️ Informacje o kolumnach (typy danych, liczba nie-null):")
print(data.info())


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 2: FEATURE ENGINEERING — LEPSZE CECHY
# ─────────────────────────────────────────────────────────────
# Tutaj tworzymy:
#  - Age              → wiek auta,
#  - Mileage_per_year → przebieg roczny (Mileage podzielony przez wiek),
#  - log_mileage      → logarytm przebiegu (żeby „przyciąć” bardzo duże wartości),
#  - log_price        → logarytm ceny (na tym trenujemy model).
#
# Uwaga: NIE tworzymy cech opartych bezpośrednio na Price jako wejściu modelu
# (np. Price_per_year, Price/Engine), bo to powodowałoby data leakage
# (model „podglądałby” odpowiedź w cechach).

# Wiek auta
data["Age"] = CURRENT_YEAR - data["Year"]

# Drobne zabezpieczenie: jeśli jakieś Year jest nielogiczne (np. > CURRENT_YEAR),
# Age może wyjść ujemny – w praktyce można by to potem „przyciąć” do min. 0.
data["Age"] = data["Age"].clip(lower=0)

# Przebieg na rok – Mileage_per_year = Mileage / (Age + 1)
# +1 w mianowniku → żeby uniknąć dzielenia przez zero przy nowych autach (Age=0).
data["Mileage_per_year"] = data["Mileage"] / (data["Age"] + 1)

# Logarytm przebiegu – żeby ograniczyć wpływ bardzo dużych przebiegów.
data["log_mileage"] = np.log1p(data["Mileage"])

# Logarytm ceny – na tym będziemy trenować model.
data["log_price"] = np.log1p(data["Price"])

print("\n✅ ROZDZIAŁ 2: Dodane kolumny 'Age', 'Mileage_per_year', 'log_mileage', 'log_price':")
print(data[["Year", "Age", "Mileage", "Mileage_per_year", "log_mileage", "Price", "log_price"]].head())


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 3: WYBÓR CECH (X) I ZMIENNEJ CELU (y)
# ─────────────────────────────────────────────────────────────
# Cechy numeryczne:
#  - Mileage           → surowy przebieg,
#  - Engine Size       → pojemność silnika,
#  - Age               → wiek auta,
#  - Mileage_per_year  → przebieg roczny,
#  - log_mileage       → transformacja logarytmiczna przebiegu.
#
# Cechy kategoryczne:
#  - Fuel Type, Brand, Condition
#
# Zmienna celu:
#  - y      → Price (oryginalna skala, do metryk),
#  - y_log  → log_price (na tym trenujemy model).

numeric_features = ["Mileage", "Engine Size", "Age", "Mileage_per_year", "log_mileage"]
categorical_features = ["Fuel Type", "Brand", "Condition"]

X = data[numeric_features + categorical_features]
y = data["Price"]
y_log = data["log_price"]

print("\n📦 ROZDZIAŁ 3: X, y przygotowane.")
print("Cechy numeryczne:", numeric_features)
print("Cechy kategoryczne:", categorical_features)


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 4: PODZIAŁ DANYCH NA ZBIORY TRENINGOWY I TESTOWY
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
    X,
    y_log,   # y do trenowania (log_price)
    y,       # y w oryginalnej skali (Price) do metryk
    test_size=0.2,
    random_state=42
)

print("\n✂️ ROZDZIAŁ 4: Podział na train/test wykonany.")
print("Rozmiar X_train:", X_train.shape)
print("Rozmiar X_test:", X_test.shape)


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 5: PREPROCESSING — IMPUTACJA, SKALOWANIE, ONE-HOT
# ─────────────────────────────────────────────────────────────

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

print("\n🛠 ROZDZIAŁ 5: Preprocessing zbudowany (ColumnTransformer).")


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 6: DEFINICJA MODELI DO PORÓWNANIA
# ─────────────────────────────────────────────────────────────

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

print("\n📚 ROZDZIAŁ 6: Zdefiniowano modele:", list(models.keys()))


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 7: TRENING MODELI I LICZENIE METRYK
# ─────────────────────────────────────────────────────────────

results = []
trained_pipelines = {}

for name, model in models.items():
    print(f"\n🚀 ROZDZIAŁ 7: Trenuję model: {name}")

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Trening na log_price
    clf.fit(X_train, y_train_log)

    trained_pipelines[name] = clf

    # Przewidywanie w skali log_price
    y_pred_log = clf.predict(X_test)

    # Powrót do skali ceny: Price = exp(log_price) - 1
    y_pred = np.expm1(y_pred_log)

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

print("\n📊 ROZDZIAŁ 7: Porównanie modeli (posortowane po RMSE):")
print(results_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# ROZDZIAŁ 8 (OPCJONALNIE): ZAPIS NAJLEPSZEGO MODELU DO PLIKU .PKL
# ─────────────────────────────────────────────────────────────
"""
import joblib
import os

os.makedirs("models", exist_ok=True)

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

MODEL_PATH = "models/best_model_advanced.pkl"
joblib.dump(best_pipeline, MODEL_PATH)

print(f"💾 ROZDZIAŁ 8: Najlepszy model: {best_model_name} zapisany jako: {MODEL_PATH}")
"""


# ─────────────────────────────────────────────────────────────
# BLOK GŁÓWNY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pass
