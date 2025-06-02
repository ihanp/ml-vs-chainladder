import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_comparison(predicted_ultimate, test_df):
    # Load Chain Ladder results
    cl_pred_ultimate = pd.read_csv("data/cl_pred_ultimate.csv", index_col=0).squeeze()

    # Step 1: Aggregate ML predictions by policy year
    ml_df = pd.DataFrame({
        "policy_year": test_df["policy_year"],
        "ml_predicted_ultimate": predicted_ultimate
    })
    ml_agg = ml_df.groupby("policy_year")["ml_predicted_ultimate"].sum()

    # Step 2: Plot Chain Ladder vs ML
    plt.figure(figsize=(12, 6))
    plt.plot(cl_pred_ultimate.index, cl_pred_ultimate.values, label="Chain Ladder", marker='o')
    plt.plot(ml_agg.index, ml_agg.values, label="ML Predicted", marker='x')
    plt.title("Ultimate Claims Prediction: Chain Ladder vs ML")
    plt.xlabel("Policy Year")
    plt.ylabel("Total Ultimate Claims")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
