# Feature Engineering

## Baseline Performance

**Reference scores:**
- LightGBM: Train 0.8288, Val 0.7610, Public 0.74255
- Logistic Regression: Train 0.7449, Val 0.7455, Public 0.73674

**Target:** Improve public score through incremental feature engineering

**Strategy:** Phase-by-phase feature addition with filtering and MLflow tracking

---

## Experiments Log

### Phase 1: Bureau Features

**Data Source:** bureau.csv (1.7M records, 305K unique customers)

**Features Created:** 52 total
- Numerical: 27 (DAYS_CREDIT, OVERDUE, AMT_CREDIT_SUM, etc.)
- Categorical: 19 (CREDIT_ACTIVE, CREDIT_TYPE via OneHotEncoder)
- Derived: 6 (debt_credit_ratio, overdue_debt_ratio, etc.)

**Filtering:**
- Level 1: Missing >80%, Variance <0.01, Correlation >0.95
- Result: 52 → 36 features
- Level 2: Importance threshold = 20 (LightGBM + XGBoost union)
- Result: 36 → 48 features (24 baseline + 24 bureau)

**Results:**
- LightGBM: Val 0.7665 (+0.0055), CV 0.7543 ± 0.0025
- XGBoost: Val 0.7582 (-0.0028), CV 0.7453 ± 0.0024

**Kaggle Scores:**
- LightGBM: Private 0.75348, Public 0.74795
- XGBoost: Private 0.74736, Public 0.73786

---

### Phase 2: Bureau Balance Features

**Base:** Phase 1 features (48)

**Data Source:** bureau_balance.csv (monthly credit status)

**Features Created:** ~17-20 total
- MONTHS_BALANCE aggregations (min, max, size)
- STATUS categories (via OneHotEncoder)
- Merged through bureau.csv → SK_ID_CURR

**Filtering:**
- Level 1: Missing >80%, Variance <0.01, Correlation >0.95
- Level 2: Importance threshold = 20
- Result: 48 features total (46 Phase 1 + 2 new bureau_balance)

**Results:**
- LightGBM: Val 0.7664 (+0.0012 from Phase 1), CV 0.7550 ± 0.0026
- XGBoost: Val 0.7591 (+0.0009 from Phase 1), CV 0.7460 ± 0.0025

**Kaggle Scores:**
- LightGBM: Private 0.75467, Public 0.74934
- XGBoost: Private 0.74857, Public 0.73306

---

### Phase 3: Previous Application Features

**Data Source:** previous_application.csv (approval history)

**Features Created:** 40 total
- Numerical: AMT_ANNUITY, AMT_GOODS_PRICE, CNT_PAYMENT, DAYS_DECISION, HOUR_APPR_PROCESS_START
- Categorical: NAME_CONTRACT_STATUS, NAME_CONTRACT_TYPE
- Derived: approval_rate, app_credit_diff, app_count

**Filtering:**
- Level 1: Variance filtering (1 dropped), Correlation filtering (7 dropped)
- Result: 40 → 32 features
- Level 2: Importance threshold = 20 (union strategy)
- Result: 55 features total (40 Phase 2 + 15 new previous application)

**Results:**
- LightGBM: Val 0.7720, CV 0.7710 ± 0.0026
- XGBoost: Val 0.7652, CV 0.7641 ± 0.0028

**Kaggle Scores:**
- LightGBM: Private 0.76267, Public 0.75670
- XGBoost: Private 0.75218, Public 0.74200

---

### Phase 4: POS & Credit Card Features

**Data Source:** POS_CASH_balance.csv + credit_card_balance.csv

**Features Created:** 54 total
- POS: MONTHS_BALANCE, CNT_INSTALMENT, CNT_INSTALMENT_FUTURE, SK_DPD, contract status
- Credit Card: AMT_BALANCE, AMT_CREDIT_LIMIT_ACTUAL, AMT_DRAWINGS, AMT_PAYMENT, SK_DPD
- Derived: balance_limit_ratio, payment_balance_ratio

**Filtering:**
- Level 1: Missing threshold (6 dropped), Variance filtering (5 dropped), Correlation filtering (10 dropped)
- Result: 54 → 33 features
- Level 2: Importance threshold = 20 (union strategy)
- Result: 76 features total (55 Phase 3 + 21 new POS/CC)

**Results:**
- LightGBM: Val 0.7735, CV 0.7716 ± 0.0033
- XGBoost: Val 0.7646, CV 0.7634 ± 0.0031

**Kaggle Scores:**
- LightGBM: Private 0.76491, Public 0.75697
- XGBoost: Private 0.75626, Public 0.75002

---

### Phase 5: Installments Features

**Data Source:** installments_payments.csv

**Features Created:** 30 total
- Numerical: AMT_INSTALMENT, AMT_PAYMENT, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT
- Derived: payment_delay, payment_diff, payment_ratio, late_payment_count, late_payment_ratio

**Filtering:**
- Level 1: No missing/variance drops, Correlation filtering (11 dropped)
- Result: 30 → 19 features
- Level 2: Importance threshold = 20 (union strategy)
- Result: 95 features total (76 Phase 4 + 19 new installments)

**Results:**
- LightGBM: Val 0.7765, CV 0.7755 ± 0.0026
- XGBoost: Val 0.7677, CV 0.7674 ± 0.0033

**Kaggle Scores:**
- LightGBM: Private 0.76940, Public 0.76412
- XGBoost: Private 0.76028, Public 0.75618

---

## Performance Tracking

| Phase | Features Added | Total Features | Val AUC (LGB) | CV Mean ± Std | Improvement from Previous |
|-------|----------------|----------------|---------------|---------------|---------------------------|
| Baseline | 36 baseline | 36 | 0.7610 | - | - |
| Phase 1 | +24 bureau | 48 | 0.7664 | 0.7543 ± 0.0025 | +0.0054 |
| Phase 2 | +2 bureau_balance | 48 | 0.7664 | 0.7549 ± 0.0022 | +0.0006 |
| Phase 3 | +15 previous_app | 55 | 0.7720 | 0.7710 ± 0.0026 | +0.0161 |
| Phase 4 | +21 pos_cc | 76 | 0.7735 | 0.7716 ± 0.0033 | +0.0007 |
| Phase 5 | +19 installments | 95 | 0.7765 | 0.7755 ± 0.0026 | +0.0039 |

**Total Improvement:** Baseline 0.7610 → Phase 5 0.7765 (+0.0155 Val AUC, +0.0212 CV AUC)

---

## Final Feature Selection

**Phase 5 Final Features:** 95 features

**Composition:**
- Baseline application features: 21 features
- Phase 1 (Bureau): 24 features
- Phase 2 (Bureau Balance): 2 features
- Phase 3 (Previous Application): 15 features
- Phase 4 (POS & Credit Card): 21 features
- Phase 5 (Installments): 19 features

**Top Contributing Feature Groups:**
1. External credit scores (EXT_SOURCE_1/2/3)
2. Bureau credit history (DAYS_CREDIT patterns, AMT_CREDIT_SUM aggregations)
3. Previous application patterns (approval_rate, app_credit_diff, DAYS_DECISION)
4. Payment behavior (late_payment_count, payment_delay, DPD metrics)
5. Credit utilization (balance_limit_ratio, payment_balance_ratio)

**Selection Method:**
- Two-level filtering: Statistical (missing/variance/correlation) + Model-based (importance threshold=20)
- LightGBM + XGBoost gain-based importance
- Union strategy for feature selection

---

## Kaggle Submission Summary

| Submission | Val AUC | CV AUC | Public Score | Private Score | Notes |
|------------|---------|--------|--------------|---------------|-------|
| Baseline Logistic | 0.7455 | - | 0.73674 | 0.72833 | 36 features |
| Baseline LightGBM | 0.7610 | - | 0.74255 | 0.74149 | 36 features |
| Phase 1 LightGBM | 0.7664 | 0.7543 | 0.74795 | 0.75348 | 48 features |
| Phase 1 XGBoost | 0.7582 | 0.7453 | 0.73786 | 0.74736 | 48 features |
| Phase 2 LightGBM | 0.7664 | 0.7549 | 0.74934 | 0.75467 | 48 features |
| Phase 2 XGBoost | 0.7591 | 0.7458 | 0.73306 | 0.74857 | 48 features |
| Phase 3 LightGBM | 0.7720 | 0.7710 | 0.75670 | 0.76267 | 55 features |
| Phase 3 XGBoost | 0.7652 | 0.7641 | 0.74200 | 0.75218 | 55 features |
| Phase 4 LightGBM | 0.7735 | 0.7716 | 0.75697 | 0.76491 | 76 features |
| Phase 4 XGBoost | 0.7646 | 0.7634 | 0.75002 | 0.75626 | 76 features |
| Phase 5 LightGBM | 0.7765 | 0.7755 | 0.76412 | 0.76940 | 95 features |
| Phase 5 XGBoost | 0.7677 | 0.7674 | 0.75618 | 0.76028 | 95 features |

**Best Model:** Phase 5 LightGBM (Private 0.76940, Public 0.76412)

---

## Key Insights

1. **LightGBM Progression:**
   - Consistent improvement: CV 0.7543 → 0.7755 (+0.0212)
   - Kaggle: Public 0.74255 → 0.76412 (+0.0216)
   - Private score (0.76940) validated CV performance
   - Stable variance across folds (std 0.0025-0.0033)

2. **Phase Impact:**
   - Phase 3 (Previous Application): Largest gain (+0.0161 CV AUC)
   - Phase 5 (Installments): Second largest (+0.0039 CV AUC)
   - Phase 1 (Bureau): Strong foundation
   - Phase 2/4: Minimal incremental gains

3. **Feature Engineering Success:**
   - Two-level filtering prevented feature explosion
   - Importance threshold (20) kept meaningful features
   - Final 95 features from 200+ candidates

4. **High-Value Patterns:**
   - Approval history highly predictive
   - Payment discipline strong signal
   - Credit utilization ratios valuable
   - Bureau credit history foundational

---

## Methodology & Technical Strategy

### Feature Creation

**Aggregations:**
- Numerical: min, max, mean, std, sum
- Categorical: OneHotEncoder expansion
- Level: All aggregated to SK_ID_CURR

**Derived Features:**
- Ratios: debt_credit_ratio, balance_limit_ratio
- Differences: app_credit_diff, payment_diff
- Counts: late_payment_count
- Rates: approval_rate, late_payment_ratio

**Data Linking:**
- Multi-level: bureau_balance → bureau → application
- Previous history via SK_ID_PREV

### Feature Filtering

**Level 1 (Statistical):**
- Missing: >80% drop
- Variance: <0.01 drop
- Correlation: >0.95 drop (keep higher importance)

**Level 2 (Model-Based):**
- Preliminary LightGBM + XGBoost training
- Gain-based importance extraction
- Threshold: 20
- Union strategy for feature selection

**Retention:** ~40-55% of created features

### Modeling

**LightGBM (Primary):**
- n_estimators: 500, learning_rate: 0.05
- max_depth: 7, num_leaves: 31
- class_weight: balanced
- Categorical: Manual integer mapping

**XGBoost (Secondary):**
- n_estimators: 500, learning_rate: 0.05
- max_depth: 7, scale_pos_weight: auto
- Used for feature importance union strategy

### Cross-Validation

**Hybrid Approach:**
1. Quick 80/20 split for fast feedback
2. 5-fold StratifiedKFold for robust estimate
3. Stratified to maintain class distribution

**Metrics:**
- Primary: ROC-AUC
- Secondary: Precision, Recall, F1
- Train-Val gap monitoring

**Persistence:**
- Saves: X_train, X_val, X_test, y_train, y_val
- Metadata: feature_metadata.json with lineage
- SK_ID_CURR preserved

### MLflow Tracking

**Parameters:**
- Phase, base phase, model hyperparameters
- Feature counts, importance threshold

**Metrics:**
- Quick: train_auc, val_auc
- CV: cv_mean_auc, cv_std_auc
- Improvement from previous phase

**Artifacts:**
- Trained models with signatures
- Feature importance, selected/dropped features (JSON)

**Organization:**
- Experiment: feature_engineering
- Runs: phaseX_source_model

