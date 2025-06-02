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

    cl_pred_ultimate = {}
    
    for year, row in triangle.iterrows():
        for t in reversed(range(max_dev + 1)):  # from dev_9 down to dev_0
            col = f"dev_{t}"
            if pd.notna(triangle_filled.loc[year, col]):
                last_value = triangle_filled.loc[year, col]
                if t == max_dev:
                    cl_pred_ultimate[year] = last_value  # fully observed
                else:
                    cumulative_factor = np.prod(factors[t:])  # include F_t onward
                    cl_pred_ultimate[year] = last_value * cumulative_factor
                break

    # Save predictions
    cl_pred_ultimate.to_csv("data/cl_pred_ultimate.csv")

    # Also print the dev factors clearly
    print("\nChain Ladder Development Factors:")
    for i, f in enumerate(factors):
        print(f"F_{i} (dev_{i} → dev_{i+1}): {f:.5f}")

    return cl_pred_ultimate
