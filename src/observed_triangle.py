import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def create_observed_triangle(df, max_dev=9, current_year=2025):
    """
    Returns a triangle DataFrame (policy_year × dev) with missing devs for later policy years.
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

    return triangle


def plot_average_development_curve(triangle, max_dev=9):
    """
    Plots average cumulative paid vs development year and returns development factors.
    """
    # Compute average cumulative curve
    avg_curve = triangle.mean(skipna=True).values

    # Compute development factors
    dev_factors = []
    for i in range(max_dev):
        if avg_curve[i] == 0 or np.isnan(avg_curve[i]):
            factor = np.nan
        else:
            factor = avg_curve[i + 1] / avg_curve[i]
        dev_factors.append(factor)

    # Plot average curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(max_dev + 1), avg_curve, marker='o')
    plt.title("Average Cumulative Development Curve")
    plt.xlabel("Development Year")
    plt.ylabel("Cumulative Paid Amount")
    plt.grid(True)
    st.pyplot(plt)

    # Return dev factors
    factor_labels = [f"dev_{i} → dev_{i+1}" for i in range(max_dev)]
    return pd.Series(dev_factors, index=factor_labels, name="Development Factors")
