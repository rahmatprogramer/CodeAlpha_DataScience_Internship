# CodeAlpha_IrisClassification

Iris Flower Classification — CodeAlpha Data Science Internship (Task 1)

## Overview
Classifies Iris flowers into three species (setosa, versicolor, virginica) based on sepal and petal measurements.

## Dataset
Classic Iris dataset (150 samples, 4 features), loaded directly via `scikit-learn` — no download needed.

## Approach
1. Loaded and explored the data (class balance, summary stats)
2. Visualized feature relationships (pairplot, boxplot by species, correlation heatmap)
3. Split into train/test sets (75/25, stratified by species)
4. Scaled features and trained **three classifiers**: Logistic Regression, K-Nearest Neighbors, and Decision Tree
5. Compared accuracy across all three models
6. Plotted a confusion matrix for the best-performing model

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 97.4% |
| K-Nearest Neighbors | 97.4% |
| Decision Tree | 97.4% |

**Key insight:** Petal length and petal width are far more discriminative than sepal measurements — setosa is perfectly separable from the other two species using petal features alone, while versicolor/virginica have some overlap.

## Files
```
iris_classification.py   # main script
output/                  # generated charts (pairplot, boxplot, heatmap, model comparison, confusion matrix)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python iris_classification.py
```

## Tech stack
Python, Pandas, Matplotlib, Seaborn, Scikit-learn (Logistic Regression, KNN, Decision Tree)

---
*Part of the CodeAlpha Data Science Internship*
