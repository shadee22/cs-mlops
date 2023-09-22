import logging
import pandas as pd
from zenml import step 
from typing_extensions import Annotated

class IngestData:
    """
    this will ingest the data from data_path
    """
    def __init__(self , data_path)  :
        """
        Args:
            data_path:  path to the data available
        """
        self.data_path = data_path

    def get_data(self):
        """
            ingest the data from data_path
        """
        logging.info(f'ingesting data from {self.data_path}') 
        data =  pd.read_csv(self.data_path)
        return data

    


@step
def ingesting_data(data_path: str) -> Annotated[pd.DataFrame, "Ingested Data"]:
    """
    this will ingest the data from data_path

    Args:
        data_path: path to the data

    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingested_data = IngestData(data_path).get_data()
        return ingested_data
    except Exception as e:
        logging.error(f'Error while ingesting data: {e}')
        raise e
