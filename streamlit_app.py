import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")

    payment_value = st.number_input("Payment Value")
    price = st.number_input("Price")
    freight_value = st.number_input("freight_value")
    product_name_length = st.number_input("Product name length")
    product_description_length = st.number_input("Product Description length")
    product_photos_qty = st.number_input("Product photos Quantity ")
    product_weight_g = st.number_input("Product weight measured in grams")
    product_length_cm = st.number_input("Product length (CMs)")
    product_height_cm = st.number_input("Product height (CMs)")
    product_width_cm = st.number_input("Product width (CMs)")

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            # run_main()

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                pred
            )
        )
    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model."
            " The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that "
            "contributes to the target variable or contributes in predicting customer satisfaction rate."
        )


if __name__ == "__main__":
    main()
