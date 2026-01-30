# -*- coding: utf-8 -*-
"""
eda.py — Exploratory Data Analysis

This script performs basic Exploratory Data Analysis (EDA):
- Load dataset
- Preview data and column info
- Descriptive statistics
- Missing values summary
- Remove duplicates
- Fill missing values
- Correlation analysis
- Visualizations (distribution, scatterplot, boxplot, heatmap)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 1️⃣ DATA LOADING
# ==========================================================

data = pd.read_csv('data/car_price_prediction_.csv')

print('\n🔹 Preview of the first 5 rows:')
print(data.head())

print('\n🔹 Column information (data types, non-null counts):')
print(data.info())

print('\n🔹 Descriptive statistics (mean, std, min, max):')
print(data.describe())

print('\n🔹 Missing values per column:')
print(data.isnull().sum())

# ==========================================================
# 2️⃣ REMOVE DUPLICATES
# ==========================================================

data = data.drop_duplicates()

# ==========================================================
# 3️⃣ FILL MISSING VALUES
# ==========================================================

data.fillna({
    'Mileage': data['Mileage'].median(),
    'Engine Size': data['Engine Size'].median(),
    'Condition': data['Condition'].mode().iloc[0],
    'Brand': data['Brand'].mode().iloc[0]
}, inplace=True)

print('\n🔹 Data after cleaning (duplicates removed and missing values filled):')
print(data.info())

# ==========================================================
# 4️⃣ CORRELATIONS
# ==========================================================

print('\n🔹 Correlation of numerical features with price:')
print(data.corr(numeric_only=True)['Price'].sort_values(ascending=False))

print('\n🔹 Most common brands:')
print(data['Brand'].value_counts().head())

# ==========================================================
# 5️⃣ VISUALIZATIONS
# ==========================================================

# ------------------------------
# 📊 HISTOGRAM
# ------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(data['Price'], bins=40, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Number of cars')
plt.show()

# ------------------------------
# 📉 SCATTERPLOT — PRICE vs MILEAGE
# ------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Mileage', y='Price', data=data, alpha=0.5)
plt.title('Price vs Mileage')
plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.show()

# ------------------------------
# 📦 BOXPLOT — PRICE vs YEAR
# ------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='Year', y='Price', data=data)
plt.title('Price by Production Year')
plt.xticks(rotation=45)
plt.show()

# ------------------------------
# 📦 BOXPLOT — PRICE VS FUEL TYPE
# ------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x='Fuel Type', y='Price', data=data)
plt.title('Price by Fuel Type')
plt.show()

# ------------------------------
# 📊 AVERAGE PRICE — TOP 10 BRANDS
# ------------------------------
brand_prices = (
    data.groupby('Brand')['Price']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
sns.barplot(x=brand_prices.index, y=brand_prices.values)
plt.title('Average Price — Top 10 Brands')
plt.ylabel('Average Price')
plt.xlabel('Brand')
plt.xticks(rotation=45)
plt.show()

# ------------------------------
# 🔥 HEATMAP
# ------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
