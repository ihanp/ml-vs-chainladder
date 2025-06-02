import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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

def plot_average_development_curve(df, max_dev=9):
    """
    Plots average cumulative paid vs development year.
    """
    # Compute mean cumulative by development year
    avg_curve = []
    for dev in range(max_dev + 1):
        dev_col = f"dev_{dev}"
        avg = df[dev_col].mean()
        avg_curve.append(avg)

    # Plot with matplotlib
    plt.figure(figsize=(8, 4))
    plt.plot(range(max_dev + 1), avg_curve, marker='o')
    plt.title("Average Cumulative Development Curve")
    plt.xlabel("Development Year")
    plt.ylabel("Cumulative Paid Amount")
    plt.grid(True)
    st.pyplot(plt)

    return triangle
