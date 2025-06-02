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
    Plots cumulative paid vs development year for the 5 most recent policy years,
    plus the overall average curve. Also returns development factors.
    """
    plt.figure(figsize=(10, 6))

    # Get 5 most recent policy years
    recent_years = sorted(triangle.index.astype(int))[-5:]
    
    # Plot each recent policy year's curve
    for year in recent_years:
        dev_values = triangle.loc[year].values.astype(float)
        plt.plot(range(max_dev + 1), dev_values, marker='o', label=f"{int(year)}")

    # Compute and plot average curve
    avg_curve = triangle.mean(skipna=True).values
    plt.plot(range(max_dev + 1), avg_curve, marker='o', linestyle='--', color='black', label="Average")

    plt.title("Development Curves: Recent Policy Years vs. Average")
    plt.xlabel("Development Year")
    plt.ylabel("Cumulative Paid Amount")
    plt.grid(True)
    plt.legend(title="Policy Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

    # Compute development factors from average curve
    dev_factors = []
    for i in range(max_dev):
        if avg_curve[i] == 0 or np.isnan(avg_curve[i]):
            factor = np.nan
        else:
            factor = avg_curve[i + 1] / avg_curve[i]
        dev_factors.append(factor)

    factor_labels = [f"dev_{i} → dev_{i+1}" for i in range(max_dev)]
    return pd.Series(dev_factors, index=factor_labels, name="Development Factors")
