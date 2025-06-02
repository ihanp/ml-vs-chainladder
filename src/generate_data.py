import numpy as np
import pandas as pd
import os

def generate_synthetic_contracts(n_contracts=100000, seed=42):
    np.random.seed(seed)
    base_curve = np.array([0.25, 0.45, 0.60, 0.70, 0.78, 0.85, 0.90, 0.94, 0.97, 1.00])
    data = []

    for i in range(n_contracts):
        contract_id = f"C{i:05d}"
        policy_year = np.random.randint(2010, 2026)

        # Create a realistic development curve
        noise = np.random.normal(loc=1.0, scale=0.05, size=10)
        dev_pattern = np.maximum.accumulate(base_curve * noise)
        dev_pattern = np.clip(dev_pattern, 0, 1)

        # Scale a random base ultimate and then define the true ultimate as dev_9
        base_ultimate = np.random.uniform(5000, 50000)
        cumulative_paid = np.round(base_ultimate * dev_pattern)
        ultimate = cumulative_paid[-1]  # Set ultimate to last dev

        row = {
            "contract_id": contract_id,
            "policy_year": policy_year,
            "ultimate": ultimate
        }
        for dev in range(10):
            row[f"dev_{dev}"] = cumulative_paid[dev]
        data.append(row)

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/all_contracts.csv", index=False)
    print("Saved data/all_contracts.csv")
    return df
