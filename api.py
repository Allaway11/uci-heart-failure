from typing import Optional, List, Dict

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from train import ClassifierModel


class PredictRequest(BaseModel):
    features: Dict[str, float]


class ModelResponse(BaseModel):
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


app = FastAPI()
# followed FastApi documentation for events on startup https://fastapi.tiangolo.com/advanced/events/
model_dict = {}


@app.on_event("startup")
async def train_model():
    clf = ClassifierModel()
    clf.train_model()
    model_dict["model"] = clf


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
    response = [{"prediction": p} for p in model_dict["model"].predict(data)]
    return ModelResponse(
        predictions=response
    )


if __name__ == "__main__":
    # To enable debugging this entrypoint to the uvicorn server has been created
    uvicorn.run(app, host="0.0.0.0", port=8000)
