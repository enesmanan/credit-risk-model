from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "final" / "lightgbm_reduced.pkl"
FEATURES_PATH = BASE_DIR / "models" / "final" / "selected_features.json"

# Model settings
PROBABILITY_THRESHOLD = 0.50
MAX_RISK_SCORE = 1.0
MIN_RISK_SCORE = 0.0

# Business rules
RISK_LEVELS = {
    "low": (0.0, 0.3),
    "medium": (0.3, 0.6),
    "high": (0.6, 1.0)
}

RISK_MESSAGES = {
    "low": "Application approved - Low risk customer",
    "medium": "Manual review required - Medium risk",
    "high": "Application rejected - High risk customer"
}

# API settings
API_TITLE = "Credit Risk Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Predict credit default risk using LightGBM model"

