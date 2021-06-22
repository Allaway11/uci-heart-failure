import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC


class ClassifierModel:
    def __init__(self):
        self.model = None

    def predict(self, features):
        return self.model.predict(features)

    def train_model(self):
        np.random.seed(2021)

        x = pd.read_json("data/train.json")

        y = x.pop("DEATH_EVENT").to_list()

        """
        followed: https://stackoverflow.com/a/48673850
        """

        numeric_features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine",
                            "serum_sodium", "time"]
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_features = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        ensemble = VotingClassifier(estimators=[
            ("rfc", RandomForestClassifier()),
            ("svm", SVC()),
            ("mlp", MLPClassifier())
        ])

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('ensemble', ensemble)
        ])

        clf.fit(x, y)
        self.model = clf

    def benchmark(self):
        if not self.model:
            return "Must train model before benchmark: call train_model() first."

        x_test = pd.read_json("data/test.json")

        y_test = x_test.pop("DEATH_EVENT").to_list()

        predictions = self.model.predict(x_test)

        return classification_report(y_true=y_test, y_pred=predictions)


if __name__ == "__main__":
    classifier = ClassifierModel()
    classifier.train_model()
    print(classifier.benchmark())
