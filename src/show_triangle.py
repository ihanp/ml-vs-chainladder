# src/show_prediction_triangle.py

import pandas as pd
import numpy as np

def get_prediction_triangle(data_path="data/all_contracts.csv", current_year=2024, max_dev=9):
    df = pd.read_csv(data_path)
    test_df = df[df["policy_year"] > 2014].copy()

    triangle = pd.DataFrame(columns=[f"dev_{i}" for i in range(max_dev + 1)])

    for year in sorted(test_df["policy_year"].unique()):
        observed_dev = min(current_year - year, max_dev + 1)
        row = test_df[test_df["policy_year"] == year][[f"dev_{i}" for i in range(observed_dev)]].sum()
        # Pad with NaNs to make it full width
        full_row = np.concatenate([row.values, [np.nan] * (max_dev + 1 - observed_dev)])
        triangle.loc[year] = full_row

    return triangle
