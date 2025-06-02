import pandas as pd
import numpy as np

def create_observed_triangle(df, max_dev=9, current_year=2025):
    """
    Returns a triangle DataFrame (policy_year × dev) with missing devs for later policy years.
    Assumes input df contains 'policy_year' and 'dev_0' to 'dev_9' columns.
    """
    triangle = pd.DataFrame(columns=[f"dev_{i}" for i in range(max_dev + 1)])

    for year in sorted(df["policy_year"].unique()):
        rows = df[df["policy_year"] == year]
        devs = rows[[f"dev_{i}" for i in range(max_dev + 1)]].sum().values

        # Determine how many devs should be observed
        observed_devs = max(0, min(current_year - year + 1, max_dev + 1))  # +1 for dev_0
        observed_row = [
            devs[i] if i < observed_devs else np.nan
            for i in range(max_dev + 1)
        ]
        triangle.loc[year] = observed_row

    return triangle
