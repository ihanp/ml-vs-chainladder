# src/show_triangle.py

import pandas as pd

def get_training_triangle(data_path="data/all_contracts.csv", max_dev=9):
    df = pd.read_csv(data_path)
    train_df = df[df["policy_year"] <= 2014].copy()
    triangle = pd.DataFrame(columns=range(max_dev + 1))

    for year in sorted(train_df["policy_year"].unique()):
        rows = train_df[train_df["policy_year"] == year]
        dev = rows[[f"dev_{i}" for i in range(max_dev + 1)]].sum().values
        triangle.loc[year] = dev

    return triangle
