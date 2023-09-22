import logging
import pandas as pd
from zenml import step
from src.cleaning_data import DataStrategy, DataCleaning, DataPreProcessingStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],
]:
    """
    Cleans and preprocesses input data.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing preprocessed data:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.DataFrame): Training labels.
            - y_test (pd.DataFrame): Testing labels.

    Raises:
        Exception: If an error occurs during data cleaning.
    """
    try:
        # Data Preprocessing
        logging.info("Data Preprocessing Started")
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        logging.info("Data Preprocessing Finished")

        # Data Splitting
        logging.info("Data Split Started")
        split_strategy = DataSplitStrategy()
        data_splitting = DataCleaning(processed_data, split_strategy)
        logging.info("Data Split Finished")
        X_train, X_test, y_train, y_test = data_splitting.handle_data()
        logging.info("Data Cleaning Finished and X_train, X_test, y_train, y_test Created")

        # Convert to DataFrames
        X_train_df, X_test_df, y_train_df, y_test_df = (
            pd.DataFrame(X_train),
            pd.DataFrame(X_test),
            pd.DataFrame(y_train),
            pd.DataFrame(y_test),
        )
        return X_train_df, X_test_df, y_train_df, y_test_df
    except Exception as e:
        logging.error(f'Error while cleaning data: {e}')
        raise e




