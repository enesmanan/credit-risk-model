import requests
import json
from pathlib import Path

# Load features
base_dir = Path(__file__).parent.parent.parent
features_path = base_dir / 'models' / 'final' / 'selected_features.json'

with open(features_path, 'r') as f:
    features = json.load(f)

# Sample data
sample_data = {
    'EXT_SOURCE_1': 0.5,
    'EXT_SOURCE_3': 0.6,
    'DAYS_BIRTH': -15000,
    'AMT_CREDIT': 500000,
    'AMT_ANNUITY': 25000,
    'EXT_SOURCE_2': 0.55,
    'AMT_GOODS_PRICE': 450000,
    'bureau_DAYS_CREDIT_ENDDATE_max': 1000,
    'DAYS_EMPLOYED': -2000,
    'pos_CNT_INSTALMENT_FUTURE_mean': 10.0,
    'inst_inst_payment_delay_max': 5.0,
    'bureau_DAYS_CREDIT_max': 2000,
    'prev_app_credit_diff': 50000,
    'DAYS_ID_PUBLISH': -3000,
    'inst_AMT_PAYMENT_sum': 100000,
    'prev_approval_rate': 0.8,
    'prev_HOUR_APPR_PROCESS_START_mean': 12.0,
    'bureau_AMT_CREDIT_SUM_sum': 800000,
    'pos_CNT_INSTALMENT_FUTURE_min': 5,
    'REGION_POPULATION_RELATIVE': 0.02,
    'DAYS_REGISTRATION': -5000,
    'DAYS_LAST_PHONE_CHANGE': -1000,
    'prev_DAYS_DECISION_min': -200,
    'bureau_AMT_CREDIT_MAX_OVERDUE_mean': 0,
    'inst_AMT_PAYMENT_min': 5000,
    'prev_DAYS_DECISION_max': -50,
    'AMT_INCOME_TOTAL': 150000,
    'bureau_AMT_CREDIT_SUM_max': 500000,
    'prev_CNT_PAYMENT_std': 2.5,
    'inst_inst_payment_ratio_mean': 0.95,
    'prev_AMT_GOODS_PRICE_max': 600000,
    'prev_AMT_ANNUITY_max': 30000,
    'bureau_DAYS_CREDIT_min': -1000,
    'inst_inst_payment_diff_mean': 100,
    'bb_months_balance_size_mean': 20,
    'bureau_debt_credit_ratio': 0.3,
    'pos_status_Active': 1,
    'prev_AMT_ANNUITY_mean': 22000,
    'prev_NAME_CONTRACT_STATUS_Refused': 0,
    'prev_app_annuity_credit_ratio': 0.05
}

# Fill missing features with 0
feature_dict = {f: sample_data.get(f, 0.0) for f in features}

# Test health endpoint
print("Testing /health endpoint...")
response = requests.get('http://localhost:8000/health')
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test features endpoint
print("Testing /features endpoint...")
response = requests.get('http://localhost:8000/features')
print(f"Status: {response.status_code}")
print(f"Features count: {len(response.json()['features'])}\n")

# Test predict endpoint
print("Testing /predict endpoint...")
response = requests.post(
    'http://localhost:8000/predict',
    json={'features': feature_dict}
)
print(f"Status: {response.status_code}")
result = response.json()
print(f"Probability: {result['probability']:.4f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Message: {result['message']}")
print(f"Features Used: {result['features_used']}")

