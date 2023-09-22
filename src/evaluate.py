from abc import ABC, abstractmethod
import logging
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class Evaluate(ABC):
    """
    Abstract class that defines a method called `calculate_scores`.
    
    This class serves as a blueprint for other classes that need to implement their own version of the `calculate_scores` method.
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Abstract method to calculate evaluation scores for a model.
        
        Parameters:
        - y_true (np.ndarray): The true labels.
        - y_pred (np.ndarray): The predicted labels.
        
        Returns:
        - The evaluation scores for the model.
        """
        pass


class MSE(Evaluate):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):

        try:
            # logging.info("Calculating mse")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("An error occurred during calculating MSE: {}".format(str(e)))
            raise e


class R2Score(Evaluate):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            mse = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("An error occurred during calculating R2 Score: {}".format(str(e)))
            raise e


class Rmse(Evaluate):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            # logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("An error occurred during calculating RMSE: {}".format(str(e)))
            raise e
