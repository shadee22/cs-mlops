from pipelines.train import train_pipeline
import logging
from zenml.client import Client
from pandas import read_csv

if  __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/olist_customers_dataset.csv")

    
    
    