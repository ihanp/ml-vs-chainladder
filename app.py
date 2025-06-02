import streamlit as st
from src.generate_data import generate_synthetic_contracts
from src.prepare_data import prepare_train_test_split
from src.train_model import train_and_save_model
from src.predict_model import predict_and_evaluate
from src.chain_ladder import chain_ladder_forecast
from src.plot_results import plot_comparison

st.title("ML vs Chain Ladder: Claims Forecasting")

if st.button("Generate Synthetic Data"):
    generate_synthetic_contracts()
    st.success("Synthetic data generated.")

if st.button("Prepare Data"):
    prepare_train_test_split()
    st.success("Data prepared for training.")

if st.button("Train Model"):
    train_and_save_model()
    st.success("Model trained and saved.")

if st.button("Predict and Evaluate"):
    predict_and_evaluate()
    st.success("Prediction and evaluation completed.")

if st.button("Chain Ladder Forecast"):
    chain_ladder_forecast()
    st.success("Chain Ladder forecast completed.")

if st.button("Plot Results"):
    plot_comparison()
    st.success("Results plotted.")
