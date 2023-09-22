from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    model_name: str = "linear_regression"
    fine_tuning: bool = False
