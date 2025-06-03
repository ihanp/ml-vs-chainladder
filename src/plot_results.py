import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def plot_comparison(predicted_ultimate, test_df):
    cl_pred_ultimate = pd.read_csv("data/cl_pred_ultimate.csv", index_col=0).squeeze()
    cl_pred_ultimate.index = cl_pred_ultimate.index.astype(int)
    cl_pred_ultimate = pd.to_numeric(cl_pred_ultimate, errors='coerce')

    # Aggregate ML predictions
    ml_df = pd.DataFrame({
        "policy_year": test_df["policy_year"],
        "ml_predicted_ultimate": predicted_ultimate
    })
    ml_agg = ml_df.groupby("policy_year")["ml_predicted_ultimate"].sum()

    # Aggregate true ultimate values
    true_agg = test_df.groupby("policy_year")["ultimate"].sum()

    # Filter from 2015 onward
    cl_pred_ultimate = cl_pred_ultimate[cl_pred_ultimate.index >= 2015]
    ml_agg = ml_agg[ml_agg.index >= 2015]
    true_agg = true_agg[true_agg.index >= 2015]

    print("Chain Ladder Head:")
    print(cl_pred_ultimate.head())

    print("\nML Aggregated Head:")
    print(ml_agg.head())

    print("\nSimulated Aggregated Head:")
    print(true_agg.head())

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(cl_pred_ultimate.index, cl_pred_ultimate.values, label="Chain Ladder", marker='o')
    plt.plot(ml_agg.index, ml_agg.values, label="ML Predicted", marker='x')
    plt.plot(true_agg.index, true_agg.values, label="Simulated", marker='s', linestyle='--')
    plt.title("Ultimate Claims Prediction: Chain Ladder vs ML vs Simulated")
    plt.xlabel("Policy Year")
    plt.ylabel("Total Ultimate Claims")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)
