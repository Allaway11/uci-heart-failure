import json

import pytest
from fastapi.testclient import TestClient

from api import app


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIXTURES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@pytest.fixture(autouse=True)
def mock_clf(mocker):
    clf = mocker.Mock()
    clf.predict.return_value = [1.0]
    return clf


@pytest.fixture(autouse=True)
def mock_classifier(mocker, mock_clf):
    classifier = mocker.patch("api.ClassifierModel")
    classifier.return_value = mock_clf
    return classifier


@pytest.fixture()
def client():
    client = TestClient(app)
    return client


example_features = {
    "age": 24.0,
    "smoking": 1.0,
    "sex": 1.0
}


get_model_predictions_examples = [
    pytest.param(
        {"url": "/predict", "headers": {"Content-Type": "application/xml"},
         "data": json.dumps({"features": example_features})},
        422,  # Unprocessable Entity as endpoint is expecting a dictionary
        id="wrong_header"
    ),
    pytest.param(
        {"url": "/predict_api", "headers": {"Content-Type": "application/json"},
         "data": json.dumps({"features": example_features})},
        404,  # Not Found - Endpoint does not exist
        id="wrong_url"
    ),
    pytest.param(
        {"url": "/predict", "headers": {"Content-Type": "application/json"},
         "data": json.dumps({"no_features": {}})},
        422,  # Unprocessable Entity - Missing value
        id="no_features"
    ),
]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_root_endpoint(client):
    response = client.get("/", headers={"Content-Type": "application/json"})
    text = json.loads(response.text)
    assert not text["predictions"]
    assert text["error"] == "This is a test endpoint."
    assert response.status_code == 200


def test_get_predict_endpoint(client):
    response = client.get("/predict")
    text = json.loads(response.text)
    assert not text["predictions"]
    assert text["error"] == "Send a POST request to this endpoint with 'features' data."
    assert response.status_code == 200


@pytest.mark.parametrize("kwargs, expected", get_model_predictions_examples)
def test_post_predict_http_client_failures(client, kwargs, expected):
    response = client.post(**kwargs)
    assert response.status_code == expected


def test_post_predict(client):
    with client as c:
        response = c.post(url="/predict", headers={"Content-Type": "application/json"},
                          data=json.dumps({"features": example_features}))
        text = json.loads(response.text)
        assert text["predictions"][0]["prediction"] == 1.0
