# Baseline Models

## Overview

Two baseline models trained on Home Credit Default Risk data:
- Logistic Regression with preprocessing pipeline
- LightGBM with minimal preprocessing

## Data

**Dataset**: application_train.csv (307,511 rows, 122 columns)
**Target distribution**: 91.93% class 0, 8.07% class 1 (imbalanced)

### Feature Selection

**Total features**: 36
- Numerical: 25 features
- Categorical: 11 features

Key features include:
- Income and credit amounts (AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE)
- Days-based features (DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION)
- External data scores (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
- Credit bureau enquiries (AMT_REQ_CREDIT_BUREAU_*)
- Categorical identifiers (NAME_CONTRACT_TYPE, CODE_GENDER, NAME_INCOME_TYPE, etc.)

High missing values in EXT_SOURCE_1 (56.4%) and OCCUPATION_TYPE (31.3%).

### Train/Validation Split

**Split ratio**: 80/20 (stratified)
- Train: 246,008 samples
- Validation: 61,503 samples
- Target distribution maintained in both sets

## Model 1: Logistic Regression

### Pipeline

**Numerical preprocessing**:
- Imputation: median strategy
- Scaling: StandardScaler

**Categorical preprocessing**:
- Imputation: most_frequent strategy
- Encoding: OneHotEncoder (handle_unknown='ignore')

**Model parameters**:
- class_weight: balanced
- max_iter: 1000
- solver: lbfgs
- random_state: 42

### Results

| Metric | Train | Validation |
|--------|-------|------------|
| ROC-AUC | 0.7449 | 0.7455 |
| Precision | - | 0.1604 |
| Recall | - | 0.6755 |
| F1-Score | - | 0.2593 |

**Confusion Matrix** (Validation):
```
[[38984 17554]
 [ 1611  3354]]
```

### Top 20 Features (by absolute coefficient)

1. NAME_INCOME_TYPE_Pensioner (5.72)
2. ORGANIZATION_TYPE_XNA (5.11)
3. DAYS_EMPLOYED (4.92)
4. NAME_EDUCATION_TYPE_Academic degree (2.20)
5. NAME_INCOME_TYPE_Working (2.03)
6. NAME_INCOME_TYPE_State servant (1.95)
7. NAME_INCOME_TYPE_Commercial associate (1.94)
8. ORGANIZATION_TYPE_Trade: type 5 (1.07)
9. ORGANIZATION_TYPE_Trade: type 4 (1.06)
10. AMT_GOODS_PRICE (0.94)

## Model 2: LightGBM

### Preprocessing

**Categorical encoding**: Manual mapping
- Each categorical value mapped to integer (train set unique values)
- Applied consistently to train, validation, test sets

**No feature scaling** (tree-based model)
**No imputation needed** (LightGBM handles missing values)

### Model Parameters

- n_estimators: 500
- learning_rate: 0.05
- max_depth: 7
- num_leaves: 31
- class_weight: balanced
- random_state: 42
- eval_metric: auc

### Results

| Metric | Train | Validation |
|--------|-------|------------|
| ROC-AUC | 0.8288 | 0.7610 |
| Precision | - | 0.1746 |
| Recall | - | 0.6588 |
| F1-Score | - | 0.2760 |

**Confusion Matrix** (Validation):
```
[[41073 15465]
 [ 1694  3271]]
```

### Top 20 Features (by gain)

1. EXT_SOURCE_3 (1345)
2. EXT_SOURCE_1 (1206)
3. EXT_SOURCE_2 (1093)
4. DAYS_BIRTH (995)
5. AMT_ANNUITY (893)
6. AMT_CREDIT (882)
7. DAYS_LAST_PHONE_CHANGE (809)
8. DAYS_ID_PUBLISH (794)
9. DAYS_EMPLOYED (779)
10. DAYS_REGISTRATION (769)

## Model Comparison

| Model | Train AUC | Val AUC | Precision | Recall | F1-Score |
|-------|-----------|---------|-----------|--------|----------|
| Logistic Regression | 0.7449 | 0.7455 | 0.1604 | 0.6755 | 0.2593 |
| LightGBM | 0.8288 | 0.7610 | 0.1746 | 0.6588 | 0.2760 |

**Observations**:
- LightGBM shows higher validation AUC (0.7610 vs 0.7455)
- LightGBM has moderate train-validation gap (0.0678) indicating some overfitting
- Logistic Regression is well-generalized (minimal train-validation gap)
- Both models handle class imbalance via balanced class weights
- External data sources (EXT_SOURCE_*) are most important features

## Saved Artifacts

**Location**: models/

1. baseline_logistic_v1.pkl
   - Complete pipeline (preprocessor + model)
   - Ready for inference

2. baseline_lightgbm_v1.pkl
   - Dictionary containing:
     - model: trained LGBMClassifier
     - cat_mappings: categorical feature mappings
     - features: feature list
     - numerical_features: numerical feature names
     - categorical_features: categorical feature names

## MLflow Tracking

**Experiment name**: baseline_models

**Logged parameters**:
- Model type
- Key hyperparameters
- Number of features

**Logged metrics**:
- train_auc
- val_auc
- val_precision
- val_recall
- val_f1

## Kaggle Submissions

**Generated files**: data/submissions/

1. baseline_logistic_v1.csv
   - Predictions on test set (48,744 samples)
   - Mean prediction: 0.4331
   - Range: [0.0086, 0.9995]
   - Predictions > 0.5: 35.34%
   - **Kaggle Score**: Private 0.72833, Public 0.73674

2. baseline_lightgbm_v1.csv
   - Predictions on test set (48,744 samples)
   - Mean prediction: 0.3894
   - Range: [0.0039, 0.9659]
   - Predictions > 0.5: 30.85%
   - **Kaggle Score**: Private 0.74149, Public 0.74255