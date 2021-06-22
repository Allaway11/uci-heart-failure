## UCI-Heart-Failure

## Setup:

Clone the repository:

```bash
git clone git@github.com:Allaway11/uci-heart-failure.git
```

create a virtual environment and install packages in `requirements.txt`

```bash
python3 -m venv venv && source venv/bin/activate
```

and then,

```bash
pip install -r requirements.txt
```

The dataset being used is: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

and can be downloaded to the data directory using the following command executed in the terminal:

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv -P data/
```

To train ML models we next need to create a training dataset and a holdout test dataset. To achieve this we can use the 
`create_test_dataset.py` script to split the original dataset into a training dataset ("data/train.json") with 80% of 
the data and a holdout test set with 20% of the data (To train ML models we next need to create a training dataset and a
holdout test dataset. To achieve this we can use the `create_test_dataset.py` script to split the original dataset into 
a training dataset (data/train.json) with 80% of the data and a holdout test set with 20% of the data (data/test.json). 
An example json file to be used as part of the curl request to the model server api is also generated - this contains
the first example in the test dataset (data/test_post_request.json).

```bash
python -m create_test_dataset
```

## Model Serving

As this dataset is small (299 examples in total) only simple ML models have been chosen (Random Forests, SVM, MLPs)
in the ensemble to avoid overfitting. Due to the lightweight resource requirements of the models, training of the model 
occurs on server start up. 

The models can be trained and benchmarked by running the following command:

```bash
python -m train
```

This should result in the following output:

```text
              precision    recall  f1-score   support

           0       0.84      0.88      0.86        41
           1       0.71      0.63      0.67        19

    accuracy                           0.80        60
   macro avg       0.77      0.75      0.76        60
weighted avg       0.80      0.80      0.80        60

```

To start up the server and train the model we can run the following command from the terminal:

```bash
uvicorn api:app
```

We can then test out a post request on the "/predict" endpoint using the browser at http://localhost:8000/docs and use a request body such as 

```json
{
  "features":{
    "age":94.0,
    "anaemia":0.0,
    "creatinine_phosphokinase":582.0,
    "diabetes":1.0,
    "ejection_fraction":38.0,
    "high_blood_pressure":1.0,
    "platelets":263358.03,
    "serum_creatinine":1.83,
    "serum_sodium":134.0,
    "sex":1.0,
    "smoking":0.0,
    "time":27.0
  }
}
```

or send a curl request e.g.:
```bash
curl -X POST --header "Content-Type: application/json" -d @data/test_post_request.json http://localhost:8000/predict   
```

## Tests

To run the unit tests associated with the api run the following command in the terminal

```bash
python -m pytest test_api.py --cov=api --cov-report=term
```