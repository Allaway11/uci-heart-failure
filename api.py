from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from train import train, ClassifierModel
import pandas as pd


class PredictRequest(BaseModel):
    features: Dict[str, float]


class ModelResponse(BaseModel):
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


app = FastAPI()
clf = ClassifierModel()


@app.on_event("startup")
async def train_model():
    """https://medium.com/analytics-vidhya/serve-a-machine-learning-model-using-sklearn-fastapi-and-docker-85aabf96729b
    """
    clf.model = train()


@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(error="This is a test endpoint.")


@app.get("/predict", response_model=ModelResponse)
async def explain_api() -> ModelResponse:
    return ModelResponse(
        error="Send a POST request to this endpoint with 'features' data."
    )


@app.post("/predict")
async def get_model_predictions(request: PredictRequest) -> ModelResponse:
    # Load data into pandas dataframe from request
    data = pd.DataFrame.from_dict(data=request.features, orient="index").T
    response = [{"prediction": p} for p in clf.model.predict(data)]
    return ModelResponse(
        predictions=response
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
