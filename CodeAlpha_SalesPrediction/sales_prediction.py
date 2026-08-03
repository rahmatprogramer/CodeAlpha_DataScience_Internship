"""
TASK 4: Sales Prediction using Python
--------------------------------------
Goal: Predict Sales from advertising spend on TV, Radio, and Newspaper.

This version explores whether advertising spend has a NON-linear effect on
sales (diminishing returns) by comparing plain Linear Regression against
Ridge Regression on polynomial features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_style("darkgrid")

# -----------------------------
# STEP 1: Load & clean
# -----------------------------
df = pd.read_csv("data/Advertising.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
print("Shape:", df.shape)
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# -----------------------------
# STEP 2: Distribution of spend across channels
# -----------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, col in zip(axes, df.columns):
    sns.histplot(df[col], kde=True, ax=ax, color="teal")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("output/1_distributions.png", dpi=120)
plt.close()

# -----------------------------
# STEP 3: Total spend vs sales (combined view)
# -----------------------------
df["Total_Spend"] = df["TV"] + df["Radio"] + df["Newspaper"]
plt.figure(figsize=(7, 5))
sns.regplot(data=df, x="Total_Spend", y="Sales", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("Total Advertising Spend vs Sales")
plt.tight_layout()
plt.savefig("output/2_total_spend_vs_sales.png", dpi=120)
plt.close()

# -----------------------------
# STEP 4: Train/test split
# -----------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# -----------------------------
# STEP 5: Compare Linear Regression vs Ridge on polynomial features
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Degree-2 polynomial features let the model capture diminishing-returns
# curves (e.g. extra TV spend past a point adding less incremental sales)
poly_ridge = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0)
)
poly_ridge.fit(X_train, y_train)
poly_pred = poly_ridge.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    return r2

r2_lin = evaluate("Linear Regression", y_test, linear_pred)
r2_poly = evaluate("Ridge + Polynomial(deg=2)", y_test, poly_pred)

# 5-fold cross-validation for a more robust comparison
cv_lin = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
cv_poly = cross_val_score(poly_ridge, X, y, cv=5, scoring="r2")
print(f"\nCross-val R2 (Linear): {cv_lin.mean():.4f} (+/- {cv_lin.std():.4f})")
print(f"Cross-val R2 (Ridge+Poly): {cv_poly.mean():.4f} (+/- {cv_poly.std():.4f})")

best_name, best_pred = ("Ridge + Polynomial", poly_pred) if r2_poly > r2_lin else ("Linear Regression", linear_pred)

# -----------------------------
# STEP 6: Visualize both models' fit
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, linear_pred, alpha=0.6, label="Linear Regression")
plt.scatter(y_test, poly_pred, alpha=0.6, label="Ridge + Polynomial", marker="x")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Model Comparison: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("output/3_model_comparison.png", dpi=120)
plt.close()

# -----------------------------
# STEP 7: Residual analysis for the best model
# -----------------------------
residuals = y_test.values - best_pred
plt.figure(figsize=(7, 4))
sns.histplot(residuals, kde=True, color="purple")
plt.axvline(0, color="black", linestyle="--")
plt.title(f"Residuals Distribution — {best_name}")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig("output/4_residuals.png", dpi=120)
plt.close()

# -----------------------------
# STEP 8: Business insight
# -----------------------------
coefs = pd.Series(linear_model.coef_, index=X.columns).sort_values(ascending=False)
print("\nLinear coefficients (impact per $1000 spend):\n", coefs)
print(f"\nBest performing approach: {best_name}")
print("Business takeaway: Radio and TV spend show the clearest link to sales;")
print("Newspaper spend contributes the least and could be reduced first if trimming budget.")

print("\nDone. Charts saved in the output/ folder.")
