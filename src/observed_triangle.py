import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_observed_triangle(df, max_dev=9, current_year=2025):
    """
    Returns a triangle DataFrame (policy_year × dev) with missing devs for later policy years.
    Also plots average cumulative payment by development year.
    """
    triangle = pd.DataFrame(columns=[f"dev_{i}" for i in range(max_dev + 1)])

    for year in sorted(df["policy_year"].unique()):
        rows = df[df["policy_year"] == year]
        devs = rows[[f"dev_{i}" for i in range(max_dev + 1)]].sum().values

        observed_devs = max(0, min(current_year - year + 1, max_dev + 1))
        observed_row = [
            devs[i] if i < observed_devs else np.nan
            for i in range(max_dev + 1)
        ]
        triangle.loc[year] = observed_row

    # --- Plot: Average Cumulative Development ---
    avg_cum = triangle.mean(skipna=True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(max_dev + 1), avg_cum, marker='o')
    plt.title("Average Cumulative Payment by Development Year")
    plt.xlabel("Development Year")
    plt.ylabel("Average Cumulative Paid Amount")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return triangle
