## Credit Risk Model

Machine learning system simulating a bank's credit scoring decisions. Predicts whether a loan applicant will default or successfully repay. Built on [Kaggle's Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset to replicate real-world credit scoring infrastructure.

**🌐 [Try the Application](https://credit-risk-model-nnnt.onrender.com/)**


### Dataset & Methodology

**Data:** 307K loan applications, 7 related tables (bureau history, previous applications, payment records)
- Target: 8% default rate (imbalanced)
- Features: 122 initial → 95 engineered → 40 final

**Pipeline:**
1. Baseline models (Logistic Regression, LightGBM)
2. Incremental feature engineering (Bureau, Previous Apps, Installments, POS/CC)
3. Two-level filtering (statistical + importance-based)
4. Hyperparameter optimization (Optuna)
5. Model feature reduction (95→40 features)

**Results:**
- Validation AUC: 0.7610 (baseline) → 0.7780 (final) [+2.2%]
- Kaggle Score: Public 0.77284 | Private 0.77125

**Documentation:**
- [Setup & Installation](docs/00_setup.md)
- [Data Overview](docs/01_data_overview.md) - Dataset schema and relationships
- [Baseline Models](docs/02_baseline.md) - Baseline modeling results
- [Feature Engineering](docs/03_feature_engineering.md) - Phase-by-phase feature creation
- [Model Optimization](docs/04_model_optimization.md) - Hyperparameter tuning with Optuna
- [API Deployment](docs/api_deployment.md) - FastAPI deployment guide


### TODO
- [ ] Review Kaggle winning solutions writeups
- [ ] EDA on additional datasets
- [ ] Detailed feature engineering and model selection architecture
- [ ] Refactor repetitive code (MLflow, feature selection) into helpers
- [ ] Create final pipeline notebook after R&D phase
- [ ] Feature stability analysis (PSI monitoring)
- [ ] Business review of all features in final model
- [ ] Threshold optimization
- [ ] Scorecard conversion
- [ ] Build segmentation model for risk level tracking
- [ ] Limit calculation system design
- [ ] Add interest rate pricing to limit system
- [ ] Model monitoring dashboard (PSI-GINI)
- [ ] Add system design architecture to readme

