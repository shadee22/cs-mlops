import json
import logging
from pipelines.Utils import get_data_for_test
import os
import numpy as np
from steps.clean_data import clean_data
from steps.evaluation import evaluate
from steps.ingest_data import ingesting_data
from steps.model_train import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
import pandas as pd
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

docker_settings = DockerSettings(required_integrations=[MLFLOW])


# import os


# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.pipelines import pipeline
# from zenml.steps import BaseParameters, Output, step


requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data


class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0.9


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
        pipeline_step_name: the name of the step that deployed the MLflow prediction
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    # get the ML flow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No ML flow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )

    print(f"Service available:  {existing_services}")
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> np.ndarray:
    """Run an inference request against a prediction service"""
    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Link all the steps artifacts together
    df = ingesting_data(data_path="./data/olist_customers_dataset.csv")
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, rmse = evaluate(model, x_test, y_test)
    deployment_decision = deployment_trigger(accuracy=mse)
    logging.info(f"Started the deployment pipeline")
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
    logging.info("Finished the deployment pipeline")

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts togethe
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)


# Reloading configuration file /Users/macbook/Desktop/HOME/Ai_Projects/mlOps/customer_satisfaction/.zen/config.yaml
# Step mlflow_model_deployer_step has finished in 39.169s.
# Run continuous_deployment_pipeline-2023_09_22-06_36_10_847324 has finished in 1m33s.
# Dashboard URL: http://127.0.0.1:8237/workspaces/default/pipelines/c06b4759-be79-4b1c-99e3-cb62d4048822/runs/3f8cf2b7-8aef-46ae-97bd-e4944176eff4/dag
# You can run:
#      mlflow ui --backend-store-uri 'file:/Users/macbook/Library/Application Support/zenml/local_stores/537a1fe0-4296-4dbd-a79c-5ce4eaa98934/mlruns
#  ...to inspect your experiment runs within the ML flow UI.
# You can find your runs tracked within the `mlflow_example_pipeline` experiment. There you'll also be able to compare two or more runs.
#
#
# existing_services: [MLFlowDeploymentService[63a724d7-4ae8-4af2-b22b-77e9987c7505] (type: model-serving, flavor: mlflow)]
# The MLflow prediction server is running locally as a daemon process service and accepts inference requests at:
#     http://127.0.0.1:8009/invocations
# To stop the service, run `zenml model-deployer models delete 63a724d7-4ae8-4af2-b22b-77e9987c7505`.
# (env) (base) MacBooks-MacBook-Pro:customer_satisfaction macbook$ Starting service 'MLFlowDeploymentService[63a724d7-4ae8-4af2-b22b-77e9987c7505] (type: model-serving, flavor: mlflow)'.
# bash: Starting: command not found
# (env) (base) MacBooks-MacBook-Pro:customer_satisfaction macbook$ python run_deployment.py --config predict
# Initiating a new run for the pipeline: inference_pipeline.
# Registered new version: (version 2).
# Executing a new run.
# Caching is disabled by default for inference_pipeline.
# Using user: default
# Using stack: mlflow_stack
#   orchestrator: default
#   artifact_store: default
#   model_deployer: mlflow_customer
#   experiment_tracker: mlflow_tracker_customer


