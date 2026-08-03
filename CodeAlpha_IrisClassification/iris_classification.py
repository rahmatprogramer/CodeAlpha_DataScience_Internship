"""
TASK 1: Iris Flower Classification
--------------------------------------
Goal: Classify Iris flowers (setosa, versicolor, virginica) into their
correct species using sepal/petal measurements.

Uses the classic Iris dataset (built into scikit-learn, so no download needed —
this is the same dataset as the CodeAlpha download link).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style("whitegrid")

# -----------------------------
# STEP 1: Load the data
# -----------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nClass balance:\n", df["species"].value_counts())
print("\nSummary stats:\n", df.describe())

# -----------------------------
# STEP 2: Explore the data (EDA)
# -----------------------------
plt.figure(figsize=(8, 6))
sns.pairplot(df, hue="species", diag_kind="hist")
plt.savefig("output/1_pairplot.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="species", y="petal length (cm)")
plt.title("Petal Length by Species")
plt.tight_layout()
plt.savefig("output/2_petal_length_boxplot.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="YlGnBu")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("output/3_correlation_heatmap.png", dpi=120)
plt.close()

# -----------------------------
# STEP 3: Prepare data for modeling
# -----------------------------
X = df.drop(columns="species")
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y
)

# Scale features (helps KNN and Logistic Regression converge/compare fairly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# STEP 4: Train and compare 3 classifiers
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=1),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = (model, preds, acc)
    print(f"\n{name}: Accuracy = {acc:.4f}")
    print(classification_report(y_test, preds))

# -----------------------------
# STEP 5: Compare model accuracies
# -----------------------------
acc_series = pd.Series({name: r[2] for name, r in results.items()}).sort_values(ascending=False)
print("\nModel comparison:\n", acc_series)

plt.figure(figsize=(6, 4))
acc_series.plot(kind="bar", color=["#4C72B0", "#55A868", "#C44E52"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("output/4_model_comparison.png", dpi=120)
plt.close()

# -----------------------------
# STEP 6: Confusion matrix for the best model
# -----------------------------
best_name = acc_series.idxmax()
best_preds = results[best_name][1]

cm = confusion_matrix(y_test, best_preds, labels=iris.target_names)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/5_confusion_matrix.png", dpi=120)
plt.close()

print(f"\nBest model: {best_name} with {acc_series.max():.2%} accuracy")
print("\nDone. Charts saved in the output/ folder.")
