import logging
import pandas as pd
from zenml import step
from src.evaluate import MSE, R2Score, Rmse
from sklearn.base import RegressorMixin
from typing import Tuple
import numpy as np
from zenml.client import  Client
import mlflow
from typing_extensions import Annotated

experiment_tracker  = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
      Args:
          model: RegressorMixin
          x_test: pd.DataFrame
          y_test: pd.Series
      Returns:
          r2_score: float
          rmse: float
      """
    try:
        # prediction = model.predict(x_test)
        # evaluation = Evaluation()
        # r2_score = evaluation.r2_score(y_test, prediction)
        # mlflow.log_metric("r2_score", r2_score)
        # mse = evaluation.mean_squared_error(y_test, prediction)
        # mlflow.log_metric("mse", mse)
        # rmse = np.sqrt(mse)
        # mlflow.log_metric("rmse", rmse)

        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = Rmse()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2_score, rmse
    except Exception as e:
        logging.error(e)
        raise e