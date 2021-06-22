import pandas as pd
from sklearn.model_selection import train_test_split
from srsly import write_json

test_ratio = 0.20

data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
y = data["DEATH_EVENT"].to_list()

train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=2021, stratify=y)

train_data.to_json("data/train.json", orient="records")
test_data.to_json("data/test.json", orient="records")
example_post_request = test_data.iloc[0].to_dict()
example_post_request.pop("DEATH_EVENT")
write_json("data/test_post_request.json", {"features": example_post_request})




