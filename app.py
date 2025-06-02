import pandas as pd
import streamlit as st
from src.generate_data import generate_synthetic_contracts
from src.prepare_data import prepare_train_test_split
from src.train_model import train_and_save_model
from src.predict_model import predict_and_evaluate
from src.chain_ladder import chain_ladder_forecast
from src.plot_results import plot_comparison
from src.observed_triangle import create_observed_triangle

st.set_page_config(layout="wide")
st.title("🚗 ML vs Chain Ladder: Claims Forecasting")

if st.button("1️⃣ Generate Synthetic Data"):
    generate_synthetic_contracts()
    st.success("✅ Synthetic data generated and saved to data/all_contracts.csv")

if st.button("2️⃣ Show Observed Triangle"):
    df = pd.read_csv("data/all_contracts.csv")
    triangle = create_observed_triangle(df, current_year=2025)
    st.dataframe(triangle)

if st.button("3️⃣ Prepare Data (Train/Test + Features)"):
    prepare_train_test_split()
    st.success("✅ Data split and training pairs saved")

if st.button("4️⃣ Train ML Model"):
    train_and_save_model()
    st.success("✅ MLP model trained and saved to models/")

if st.button("5️⃣ Predict and Evaluate (Test Set)"):
    predict_and_evaluate()
    st.success("✅ Predictions generated and test metrics printed")

if st.button("6️⃣ Run Chain Ladder Forecast"):
    df = pd.read_csv("data/all_contracts.csv")
    chain_ladder_forecast(df)
    st.success("✅ Chain Ladder predictions saved")

if st.button("7️⃣ Plot ML vs Chain Ladder"):
    import numpy as np
    predicted_ultimate = np.load("data/ml_predicted_ultimate.npy")
    test_df = pd.read_csv("data/test_contracts.csv")
    plot_comparison(predicted_ultimate, test_df)
