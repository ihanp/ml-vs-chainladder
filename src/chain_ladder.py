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

    # Step 2: Compute dev factors using valid (non-NaN) dev_t → dev_{t+1} pairs
    factors = []
    for t in range(max_dev):
        curr_col = f"dev_{t}"
        next_col = f"dev_{t + 1}"

        valid = triangle[[curr_col, next_col]].dropna()
        num = valid[next_col].sum()
        denom = valid[curr_col].replace(0, np.nan).sum()
        factor = num / denom if denom != 0 else 1.0
        factors.append(factor)

    # Step 3: Forecast missing devs row-wise using factors
    triangle_filled = triangle.copy()
    for year, row in triangle.iterrows():
        for t in range(max_dev):
            curr_col = f"dev_{t}"
            next_col = f"dev_{t + 1}"
            if pd.isna(row[next_col]) and pd.notna(row[curr_col]):
                triangle_filled.loc[year, next_col] = triangle_filled.loc[year, curr_col] * factors[t]

    # Step 4: Compute predicted ultimate using cumulative factors beyond last observed dev
    cl_pred_ultimate_dict = {}
    for year, row in triangle.iterrows():
        # Find last observed development year
        last_observed_index = row.last_valid_index()
        if last_observed_index is None:
            continue
        last_dev = int(last_observed_index.split("_")[1])
        last_value = triangle_filled.loc[year, last_observed_index]

        # Multiply by remaining dev factors
        future_factors = np.prod(factors[last_dev:]) if last_dev < max_dev else 1.0
        cl_pred_ultimate_dict[year] = last_value * future_factors

    # Convert to Series and save
    cl_pred_ultimate = pd.Series(cl_pred_ultimate_dict, name="CL_predicted_ultimate")
    cl_pred_ultimate.index.name = "policy_year"
    cl_pred_ultimate.to_csv("data/cl_pred_ultimate.csv")

    # Print factors
    print("\nChain Ladder Development Factors:")
    for i, f in enumerate(factors):
        print(f"F_{i} (dev_{i} → dev_{i+1}): {f:.5f}")

    return cl_pred_ultimate
