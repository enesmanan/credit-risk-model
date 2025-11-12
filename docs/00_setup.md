# Setup

## Prerequisites

- Python 3.10.6
- UV package manager

## Installation

### Sync Dependencies

```bash
uv sync
```

### Activate Environment

```bash
.venv\Scripts\activate
```

### MLflow UI

```bash
uv run mlflow ui
```

Access at: http://localhost:5000

### Dependencies

**Core:**
- pandas, numpy, scikit-learn
- lightgbm
- mlflow
- shap
- fastapi, uvicorn, pydantic, python-multipart

**Dev:**
- jupyter, matplotlib, seaborn

## Project Structure

```
credit-risk-model/
├── data/
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned and transformed data
│   └── samples/       # Sample datasets for experiment
├── models/            # Saved model 
├── notebooks/         # Jupyter notebooks 
├── src/               # Source code 
├── docs/              # Documentation
└── mlruns/            # MLflow tracking (auto-generated)
```
