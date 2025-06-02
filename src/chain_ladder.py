import pandas as pd
import numpy as np

def chain_ladder_forecast(observed_triangle):
    """
    Applies Chain Ladder using a provided observed triangle (with NaNs).
    Saves final predicted ultimates to data/cl_pred_ultimate.csv.
    """
    max_dev = observed_triangle.shape[1] - 1
    triangle = observed_triangle.copy()

    # Step 1: Compute development factors F_t = sum(dev_{t+1}) / sum(dev_t), skipping NaNs
    factors = []
    for t in range(max_dev):
        curr_col = f"dev_{t}"
        next_col = f"dev_{t + 1}"
        valid_rows = triangle[[curr_col, next_col]].dropna()
        num = valid_rows[next_col].sum()
        denom = valid_rows[curr_col].replace(0, np.nan).sum()
        factors.append(num / denom)

    # Step 2: Forecast missing cells row-wise
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

    # Save
    cl_pred_ultimate.to_csv("data/cl_pred_ultimate.csv")
    print("Saved Chain Ladder Ultimate Predictions from observed triangle")
    return cl_pred_ultimate
