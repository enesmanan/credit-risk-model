import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from src.config import MODEL_PATH, FEATURES_PATH, RISK_LEVELS, RISK_MESSAGES


class CreditRiskPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self._model_loaded = False
    
    def load_model(self):
        if not self._model_loaded:
            self.model = joblib.load(MODEL_PATH)
            with open(FEATURES_PATH, 'r') as f:
                self.features = json.load(f)
            self._model_loaded = True
    
    def get_risk_level(self, probability: float) -> str:
        for level, (low, high) in RISK_LEVELS.items():
            if low <= probability < high:
                return level
        return "high"
    
    def predict(self, features_dict: Dict[str, float]) -> Tuple[float, str, str]:
        # Ensure model is loaded
        self.load_model()
        
        df = pd.DataFrame([features_dict])
        
        # Ensure correct feature order
        df = df[self.features]
        
        # Get prediction
        probability = self.model.predict_proba(df)[0, 1]
        risk_level = self.get_risk_level(probability)
        message = RISK_MESSAGES[risk_level]
        
        return float(probability), risk_level, message
    
    def get_feature_names(self):
        # Ensure model is loaded to get feature names
        self.load_model()
        return self.features


predictor = CreditRiskPredictor()

