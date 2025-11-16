import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict
from src.config import API_TITLE, API_VERSION, API_DESCRIPTION
from src.inference import predictor

app = FastAPI(title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION)

templates = Jinja2Templates(directory="src/templates")


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    message: str
    features_used: int


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    feature_names = predictor.get_feature_names()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "features": feature_names}
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": predictor.model is not None}


@app.get("/features")
async def get_features():
    return {"features": predictor.get_feature_names()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    probability, risk_level, message = predictor.predict(request.features)
    
    return PredictionResponse(
        probability=probability,
        risk_level=risk_level,
        message=message,
        features_used=len(request.features)
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

