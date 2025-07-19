import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/amilasnils008/mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "amilasnils008"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f34b09f6dafe0c38be253c9456fe1018d05a4b88"

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X= data.drop(columns=["Outcome"])
    y= data["Output"]
    
    ##setting mlflow tracking url:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    ##loading the model from the model 
    model = pickle.load(open(model_path),'rb')
    
    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)
    
    ##log metrics to mlflow 
    mlflow.log_metric("accuracy_score", accuracy)
    print(f"the accuracy is {accuracy}")
    
    
if __name__ == "__main__":
    evaluate(params["data"],params["model"])
    