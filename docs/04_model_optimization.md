# Model Hyperparameter Optimization

**Base Features:** Phase 5 final features (95 features)  
**Optimization Method:** Bayesian Optuna 
**Validation Strategy:** 5-fold Stratified Cross-Validation  

## Pipeline

### 1. Load Phase 5 Features
- Train: 246,008 samples
- Validation: 61,503 samples
- Test: 48,744 samples
- Features: 95 features from Phase 5

### 2. LightGBM Hyperparameter Tuning
- **Trials:** 150
- **Search Space:** 9 hyperparameters (n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda)
- **Optimization Time:** 2h 17min

### 3. XGBoost Hyperparameter Tuning
- **Trials:** 50
- **Search Space:** 9 hyperparameters (n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda)
- **Optimization Time:** 57min

## Results

### Performance Comparison

**Baseline Models:**

| Model | Train AUC | Val AUC | Public Score | Private Score |
|-------|-----------|---------|--------------|---------------|
| Logistic Regression | 0.7449 | 0.7455 | 0.73674 | 0.72833 |
| LightGBM (Initial) | 0.8288 | 0.7610 | 0.74255 | 0.74149 |

**Tuned Models (Phase 5 Features + Optuna):**

| Model | Train AUC | Val AUC | CV AUC | Public Score | Private Score |
|-------|-----------|---------|--------|--------------|---------------|
| LightGBM Tuned | 0.8223 | 0.7780 | 0.7783 | 0.77284 | 0.77125 |
| XGBoost Tuned | 0.8157 | 0.7781 | 0.7781 | 0.77056 | 0.77061 | 

### Best Hyperparameters

**LightGBM:**
```
  Params: 
    n_estimators: 981
    learning_rate: 0.0701261784638505
    max_depth: 3
    num_leaves: 50
    min_child_samples: 77
    subsample: 0.5022449878003225
    colsample_bytree: 0.9670688728186712
    reg_alpha: 0.0005469072612312737
    reg_lambda: 2.5536050141966927e-06
```

**XGBoost:**
```
  Params: 
    n_estimators: 861
    learning_rate: 0.06071618023039132
    max_depth: 3
    min_child_weight: 4
    subsample: 0.7171781358782057
    colsample_bytree: 0.865042426595599
    gamma: 0.00444970387513357
    reg_alpha: 0.0029151336209232823
    reg_lambda: 9.22101468409744e-05
```
