import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow





os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/amilasnils008/mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "amilasnils008"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f34b09f6dafe0c38be253c9456fe1018d05a4b88"


def hyperparamter_tuning(X_train,y_train,param_grid):
    
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator= rf, param_grid= param_grid, cv=3, n_jobs=-1, verbose= True)
    grid_search.fit(X_train,y_train)
    
    return grid_search
    


##loading the params from the params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


def train(data_path,model_path,random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    
    ##mlflow tracking :
    
    with mlflow.start_run():
        ##splitting the data to train and test
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42)
        signature = infer_signature(X_train,y_train)
        
        ##defining the params 
        param_grid ={
            "n_estimators" :[100,200,250],
            "max_depth" : [4,8,None],
            "min_samples_split": [2,5],
            "min_samples_leaf" : [1,2,4]
        }
        
        grid_search = hyperparamter_tuning(X_train,y_train,param_grid)
        
        ##getting the best model :
        
        best_model = grid_search.best_estimator_
        
        ##predict and evalute the model
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        
        print(f"the accuracy we got is {accuracy}")
        
        ##logging the additional metrics:
        mlflow.log_metric("accuracy",accuracy)
        
        mlflow.log_param("best_n_estimators", grid_search.best_estimator_.get_params()["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_estimator_.get_params()["max_depth"])
        mlflow.log_param("min_samples_split", grid_search.best_estimator_.get_params()["min_samples_split"])
        mlflow.log_param("min_samples_leaf", grid_search.best_estimator_.get_params()["min_samples_leaf"])
        
        ##log the confusion matrix and classification report  
        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_test,y_pred)
        
        mlflow.log_text(str(cm),"confusion matrix.txt")
        
        mlflow.log_text(cr, "classification_report.txt")
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model= best_model,
                artifact_path = "model",
                signature= signature,
                input_example= X_train,
                registered_model_name= "the best model for ml_pipeline",
            )
        else :
            mlflow.sklearn.load_model(
                sk_model = best_model,
                signature = signature
            )
        
        ##creating the directory to save the model:
        os.makedirs(os.path.dirname(model_path), exist_ok= True)
        
        
        file_name = model_path
        pickle.dump(best_model,open(file_name,"wb"))
        
        print(f"model succesfully saved to {model_path}")
        
if __name__ == "__main__":
    train(params["data"],params["model"],params["random_state"], params["n_estimators"],params["max_depth"])
        

        
        
        