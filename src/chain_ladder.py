import pandas as pd
import numpy as np

def chain_ladder_forecast(observed_triangle):
    """
    Applies Chain Ladder using a provided observed triangle (with NaNs).
    Saves final predicted ultimates to data/cl_pred_ultimate.csv.
    """
    max_dev = observed_triangle.shape[1] - 1

    # Step 1: Copy triangle
    triangle = observed_triangle.copy()

    # Step 2: Compute dev factors using available pairs only
    factors = []
    for t in range(max_dev):
        curr_col = f"dev_{t}"
        next_col = f"dev_{t + 1}"

        valid = triangle[[curr_col, next_col]].dropna()
        num = valid[next_col].sum()
        denom = valid[curr_col].replace(0, np.nan).sum()
        factor = num / denom
        factors.append(factor)

    # Step 3: Forecast missing devs row-wise
    triangle_filled = triangle.copy()
    for year, row in triangle.iterrows():
        for t in range(max_dev):
            curr_col = f"dev_{t}"
            next_col = f"dev_{t + 1}"
            if pd.isna(row[next_col]) and pd.notna(row[curr_col]):
                triangle_filled.loc[year, next_col] = triangle_filled.loc[year, curr_col] * factors[t]

    # Step 4: Compute predicted ultimate by applying remaining cumulative dev factors to last observed dev
    cl_pred_ultimate = {}
    
    for year, row in triangle.iterrows():
        observed_devs = row.first_valid_index(), row.last_valid_index()
        last_observed_index = row.last_valid_index()
        if last_observed_index is None:
            continue
        last_value = triangle_filled.loc[year, last_observed_index]
        dev_start = int(last_observed_index.split("_")[1])
        cumulative_factor = np.prod(factors[dev_start + 1:])  # Apply all remaining factors
        cl_pred_ultimate[year] = last_value * cumulative_factor
    
    cl_pred_ultimate = pd.Series(cl_pred_ultimate, name="CL_predicted_ultimate")
    cl_pred_ultimate.index.name = "policy_year"

    # Save predictions
    cl_pred_ultimate.to_csv("data/cl_pred_ultimate.csv")

    # Also print the dev factors clearly
    print("\nChain Ladder Development Factors:")
    for i, f in enumerate(factors):
        print(f"F_{i} (dev_{i} → dev_{i+1}): {f:.5f}")

    return cl_pred_ultimate
