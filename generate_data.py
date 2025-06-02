
import numpy as np
import pandas as pd

def generate_synthetic_contracts(n_contracts=100000, seed=42):
    np.random.seed(seed)

    base_curve = np.array([0.25, 0.45, 0.60, 0.70, 0.78, 0.85, 0.90, 0.94, 0.97, 1.00])
    data = []

    for i in range(n_contracts):
        contract_id = f"C{i:05d}"
        policy_year = np.random.randint(2010, 2020)  # 2010–2019
        ultimate = np.random.uniform(5000, 50000)

        # Add light multiplicative noise
        noise = np.random.normal(loc=1.0, scale=0.05, size=10)
        dev_pattern = np.maximum.accumulate(base_curve * noise)  # ensure monotonic
        dev_pattern = np.clip(dev_pattern, 0, 1)

        cumulative_paid = np.round(ultimate * dev_pattern)

        row = {
            "contract_id": contract_id,
            "policy_year": policy_year,
            "ultimate": ultimate
        }
        for dev in range(10):
            row[f"dev_{dev}"] = cumulative_paid[dev]
        
        data.append(row)

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_contracts()
    train_df = df[df["policy_year"] <= 2014].reset_index(drop=True)
    test_df = df[df["policy_year"] > 2014].reset_index(drop=True)

    train_df.to_csv("train_contracts.csv", index=False)
    test_df.to_csv("test_contracts.csv", index=False)

    print(f"✅ Saved {len(train_df)} training contracts and {len(test_df)} test contracts.")
