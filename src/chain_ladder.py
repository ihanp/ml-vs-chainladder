import pandas as pd
import numpy as np

def chain_ladder_forecast(df, observed_triangle):
    """
    Applies Chain Ladder using a provided observed triangle (with NaNs).
    Saves final predicted ultimates to data/cl_pred_ultimate.csv.
    """
    max_dev = observed_triangle.shape[1] - 1

    # Step 1: Use provided triangle instead of creating from df
    triangle = observed_triangle.copy()

    # Step 2: Compute development factors F_t = sum(dev_{t+1}) / sum(dev_t)
    complete_rows = triangle.dropna()
    factors = []
    for t in range(max_dev):
        num = complete_rows[t + 1].sum()
        denom = complete_rows[t].replace(0, np.nan).sum()
        factors.append(num / denom)

    # Step 3: Forecast missing cells row-wise
    triangle_filled = triangle.copy()
    for year, row in triangle.iterrows():
        for t in range(max_dev):
            if pd.isna(row[t + 1]):
                triangle_filled.loc[year, t + 1] = triangle_filled.loc[year, t] * factors[t]

    # Step 4: Extract dev_9 as predicted ultimate
    cl_pred_ultimate = triangle_filled[max_dev].copy()
    cl_pred_ultimate.name = "CL_predicted_ultimate"
    cl_pred_ultimate.index.name = "policy_year"

    # Save
    cl_pred_ultimate.to_csv("data/cl_pred_ultimate.csv")
    print("Saved Chain Ladder Ultimate Predictions from observed triangle")
    return cl_pred_ultimate
