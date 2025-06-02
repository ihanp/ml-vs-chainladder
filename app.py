import streamlit as st
from src.generate_data import generate_synthetic_contracts
from src.prepare_data import prepare_train_test_split
from src.train_model import train_and_save_model
from src.predict_model import predict_and_evaluate
from src.chain_ladder import chain_ladder_forecast
from src.plot_results import plot_comparison

st.set_page_config(layout="wide")
st.title("🚗 ML vs Chain Ladder: Claims Forecasting")

if st.button("1️⃣ Generate Synthetic Data"):
    generate_synthetic_contracts()
    st.success("✅ Synthetic data generated and saved to data/all_contracts.csv")

if st.button("2️⃣ Prepare Data (Train/Test + Features)"):
    prepare_train_test_split()
    st.success("✅ Data split and training pairs saved")

if st.button("3️⃣ Train ML Model"):
    train_and_save_model()
    st.success("✅ MLP model trained and saved to models/")

if st.button("4️⃣ Predict and Evaluate (Test Set)"):
    predict_and_evaluate()
    st.success("✅ Predictions generated and test metrics printed")

if st.button("5️⃣ Run Chain Ladder Forecast"):
    import pandas as pd
    df = pd.read_csv("data/all_contracts.csv")
    chain_ladder_forecast(df)
    st.success("✅ Chain Ladder predictions saved")

if st.button("6️⃣ Plot ML vs Chain Ladder"):
    import numpy as np
    import pandas as pd
    predicted_ultimate = np.load("data/ml_predicted_ultimate.npy")
    test_df = pd.read_csv("data/test_contracts.csv")
    plot_comparison(predicted_ultimate, test_df)

from src.show_triangle import get_training_triangle

# Show cumulative triangle
if st.button("📐 Show Training Triangle (Cumulative Paid)"):
    triangle = get_training_triangle()
    st.subheader("Training Triangle")
    st.dataframe(triangle)
