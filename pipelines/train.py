from zenml import pipeline
from steps.ingest_data import ingesting_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate
import logging as log

@pipeline 
def train_pipeline(data_path: str):
    ingested_data = ingesting_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(ingested_data)
    model = train_model(X_train, y_train)
    mse, r2_score, rmse = evaluate(model, X_test, y_test)