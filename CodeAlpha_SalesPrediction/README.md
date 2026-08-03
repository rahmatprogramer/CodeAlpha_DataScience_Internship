# CodeAlpha_SalesPrediction

Sales Prediction using Python — CodeAlpha Data Science Internship (Task 4)

## Overview
Predicts Sales from advertising spend on TV, Radio, and Newspaper — and specifically tests whether the relationship is non-linear (diminishing returns on ad spend).

## Dataset
`Advertising.csv` — 200 records: `TV`, `Radio`, `Newspaper` (spend in $ thousands), `Sales` (units, thousands)

## Approach
1. Loaded and explored spend distributions per channel
2. Looked at combined total spend vs. sales
3. Trained and compared two models:
   - Plain **Linear Regression**
   - **Ridge Regression on degree-2 polynomial features** (captures diminishing-returns curves)
4. Validated with a 5-fold cross-validation, not just a single train/test split
5. Analyzed residuals of the best model

## Results
| Model | MAE | RMSE | R² | CV R² (5-fold) |
|---|---|---|---|---|
| Linear Regression | 1.124 | 1.436 | 0.930 | 0.887 |
| **Ridge + Polynomial (deg=2)** | **0.424** | **0.548** | **0.990** | **0.983** |

**Key insight:** Advertising spend has a genuinely non-linear effect on sales — the polynomial model captured interaction effects (e.g., TV and Radio spend reinforcing each other) that a plain linear model misses, nearly halving the prediction error.

## Files
```
sales_prediction.py     # main script
data/Advertising.csv    # dataset
output/                 # generated charts (distributions, total spend vs sales, model comparison, residuals)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python sales_prediction.py
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (Linear Regression, Ridge, Polynomial Features)

---
*Part of the CodeAlpha Data Science Internship*
