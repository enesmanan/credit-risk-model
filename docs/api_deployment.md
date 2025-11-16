# API Deployment Guide

FastAPI application for credit risk prediction using trained LightGBM model.

## Project Structure

```
src/
├── config.py          # Paths, business rules, settings
├── inference.py       # Model loading and prediction logic
├── app.py             # FastAPI application
├── templates/
│   └── index.html     # Web interface
└── tests/
    └── test_api.py    # API tests
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn src.app:app --reload --port 8000
```

Visit: http://localhost:8000

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /features` - List required features
- `POST /predict` - Predict credit risk

## Test API

```bash
# Run tests (server must be running)
python -m src.tests.test_api
```

## Deploy to Render

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect repository
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
6. Deploy

## Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "EXT_SOURCE_1": 0.5,
      "EXT_SOURCE_3": 0.6,
      "DAYS_BIRTH": -15000,
      ...
    }
  }'
```

## Response

```json
{
  "probability": 0.234,
  "risk_level": "low",
  "message": "Application approved - Low risk customer",
  "features_used": 40
}
```

