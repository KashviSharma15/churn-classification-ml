# churn-classification-ml

# Task 1 — ML Classification: Customer Churn Prediction

**Alfido Tech ML Internship**

Predict whether a telecom customer will churn using Logistic Regression and Random Forest.
Models are compared on Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

---

## Results Summary

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | ~0.80    | ~0.64     | ~0.53  | ~0.58 | ~0.84   |
| Random Forest       | ~0.81    | ~0.66     | ~0.55  | ~0.60 | ~0.86   |

**Winner: Random Forest** — higher ROC-AUC and F1, saved as `model.pkl` for deployment.

---

## Project Structure

```
task1-churn-classification/
├── notebook/
│   └── churn_classification.ipynb   
├── data/
│   └── README.md                    
├── outputs/
│   ├── eda_plots.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── model.pkl                       
├── scaler.pkl
├── feature_names.pkl
├── requirements.txt
└── README.md
```

---


## What Each Cell Does

| Cell | What it does |
|------|-------------|
| 1    | Import all libraries |
| 2    | Load dataset, inspect shape and columns |
| 3    | Clean data — fix TotalCharges type, fill nulls, encode target |
| 4    | EDA — 4 plots: histogram, boxplot, scatter, bar chart |
| 5    | Feature engineering — encode all categorical columns |
| 6    | Train/test split (80/20, stratified) + StandardScaler |
| 7    | Train Logistic Regression, print all metrics |
| 8    | Train Random Forest, print all metrics |
| 9    | 5-fold cross-validation for both models |
| 10   | ROC curve comparison plot |
| 11   | Confusion matrix for both models |
| 12   | Feature importance bar chart |
| 13   | Final comparison table |
| 14   | Save model.pkl and scaler.pkl for Task 3 & Task 4 |

---

## Key Findings

- **Tenure** and **Contract type** are the most important features — customers on month-to-month contracts with short tenure churn far more.
- **Random Forest** outperforms Logistic Regression on all metrics.
- Class imbalance (~26% churn rate) is handled by using `stratify=y` in the split.

---

## Environment

- Python 3.11
- scikit-learn 1.5.0
- See `requirements.txt` for full list

---

*Alfido Tech Internship — Task 1 of 3*
