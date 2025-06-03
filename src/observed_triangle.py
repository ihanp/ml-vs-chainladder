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
    Plots cumulative paid vs development year for the 9 most recent policy years,
    plus the overall average curve.
    """
    import matplotlib.pyplot as plt
    import streamlit as st

    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=110)  # Smaller and sharper

    # Get 9 most recent policy years
    recent_years = sorted(triangle.index.astype(int))[-9:]
    
    # Plot recent years
    for year in recent_years:
        dev_values = triangle.loc[year].values.astype(float)
        ax.plot(range(max_dev + 1), dev_values, marker='o', markersize=3, linewidth=1, label=f"{int(year)}")

    # Average curve
    avg_curve = triangle.mean(skipna=True).values
    ax.plot(range(max_dev + 1), avg_curve, marker='o', markersize=3.5, linestyle='--', linewidth=1.5, color='black', label="Average")

    ax.set_title("Dev Curves: Recent Years vs. Avg", fontsize=8)
    ax.set_xlabel("Dev Year", fontsize=7)
    ax.set_ylabel("Cumulative Paid", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.legend(title="Policy Year", fontsize=5.5, title_fontsize=6.5, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(pad=0.6)
    st.pyplot(fig)
