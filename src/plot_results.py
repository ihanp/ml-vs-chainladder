import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def plot_comparison(predicted_ultimate, test_df):
    cl_pred_ultimate = pd.read_csv("cl_pred_ultimate.csv", index_col=0).squeeze()

    # Aggregate ML predictions
    ml_df = pd.DataFrame({
        "policy_year": test_df["policy_year"],
        "ml_predicted_ultimate": predicted_ultimate
    })
    ml_agg = ml_df.groupby("policy_year")["ml_predicted_ultimate"].sum()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(cl_pred_ultimate.index, cl_pred_ultimate.values, label="Chain Ladder", marker='o')
    plt.plot(ml_agg.index, ml_agg.values, label="ML Predicted", marker='x')
    plt.title("Ultimate Claims Prediction: Chain Ladder vs ML")
    plt.xlabel("Policy Year")
    plt.ylabel("Total Ultimate Claims")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(plt)
