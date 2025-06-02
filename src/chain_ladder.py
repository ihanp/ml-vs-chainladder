import pandas as pd
import numpy as np

def chain_ladder_forecast(observed_triangle):
    """
    Applies Chain Ladder using a provided observed triangle (with NaNs).
    Prints development factors and saves predicted ultimates.
    """
    max_dev = observed_triangle.shape[1] - 1
    triangle = observed_triangle.copy()

    # Step 1: Compute development factors, skipping only NaN pairs
    factors = []
    print("\n Chain Ladder Development Factors:")
    for t in range(max_dev):
        curr = triangle[f"dev_{t}"]
        nxt = triangle[f"dev_{t + 1}"]
        valid = (~curr.isna()) & (~nxt.isna())
        num = nxt[valid].sum()
        denom = curr[valid].replace(0, np.nan).sum()
        factor = num / denom
        factors.append(factor)
        print(f"  dev_{t} ➝ dev_{t + 1}: {factor:.4f}")

    # Step 2: Fill triangle row-wise
    triangle_filled = triangle.copy()
    for year, row in triangle.iterrows():
        for t in range(max_dev):
            curr_col = f"dev_{t}"
            next_col = f"dev_{t + 1}"
            if pd.isna(row[next_col]):
                triangle_filled.loc[year, next_col] = triangle_filled.loc[year, curr_col] * factors[t]

    # Step 3: Extract dev_9 as predicted ultimate
    cl_pred_ultimate = triangle_filled[f"dev_{max_dev}"].copy()
    cl_pred_ultimate.name = "CL_predicted_ultimate"
    cl_pred_ultimate.index.name = "policy_year"

    cl_pred_ultimate.to_csv("data/cl_pred_ultimate.csv")
    print("\n Saved Chain Ladder Ultimate Predictions to data/cl_pred_ultimate.csv")

    return cl_pred_ultimate
