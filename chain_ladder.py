import pandas as pd
import numpy as np

# Step 1: Create cumulative triangle from training data
train_df = df[df["policy_year"] <= 2014].copy()
max_dev = 9

triangle = pd.DataFrame(columns=range(max_dev + 1))

for year in sorted(train_df["policy_year"].unique()):
    rows = train_df[train_df["policy_year"] == year]
    dev = rows[[f"dev_{i}" for i in range(max_dev + 1)]].sum().values
    triangle.loc[year] = dev

# Step 2: Compute development factors F_t = sum(dev_{t+1}) / sum(dev_t)
factors = []
for t in range(max_dev):
    num = triangle[t + 1].sum()
    denom = triangle[t].replace(0, np.nan).sum()
    factors.append(num / denom)

# Step 3: Apply Chain Ladder to incomplete triangle (2015–2024)
future_years = sorted(df[df["policy_year"] > 2014]["policy_year"].unique())
cl_preds = {}

for year in future_years:
    row = df[df["policy_year"] == year]
    devs = row[[f"dev_{i}" for i in range(max_dev + 1)]].sum().values
    latest_dev = max(i for i, val in enumerate(devs) if val > 0)
    cumulative = devs.copy()

    for t in range(latest_dev, max_dev):
        cumulative[t + 1] = cumulative[t] * factors[t]
    
    cl_preds[year] = cumulative[max_dev]

# Convert to Series
cl_pred_ultimate = pd.Series(cl_preds, name="CL_predicted_ultimate")
cl_pred_ultimate.index.name = "policy_year"

# Done
print("Manual Chain Ladder Ultimate Predictions (2015–2024):")
display(cl_pred_ultimate)
