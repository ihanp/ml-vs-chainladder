import numpy as np
import pandas as pd
import os

def generate_synthetic_contracts(n_contracts=1000):
    np.random.seed(seed)
    base_curve = np.array([0.25, 0.45, 0.60, 0.70, 0.78, 0.85, 0.90, 0.94, 0.97, 1.00])
    data = []

    for i in range(n_contracts):
        contract_id = f"C{i:05d}"
        policy_year = np.random.randint(2010, 2026)
        curve = base_curve.copy()

        # 💥 Heavy random noise
        curve += np.random.normal(0, 0.1, size=10)

        # 🔁 30% chance of flip or full scramble
        p = np.random.rand()
        if p < 0.15:
            curve = curve[::-1]
        elif p < 0.30:
            curve = np.random.permutation(curve)

        # 🌀 Blend with reversed or noisy base
        if np.random.rand() < 0.3:
            noise_base = base_curve[::-1] + np.random.normal(0, 0.1, size=10)
            curve = 0.5 * curve + 0.5 * noise_base

        # ⚡ Massive shocks (5–10 shocks)
        for _ in range(np.random.randint(5, 11)):
            idx = np.random.randint(1, 9)
            shock = np.random.uniform(-0.4, 0.5)
            curve[idx:] += shock

        # 🚫 40% chance of early plateau/drop
        if np.random.rand() < 0.4:
            cutoff = np.random.randint(2, 9)
            curve[cutoff:] = curve[cutoff]

        # 🧽 Cleanup: clip and make cumulative
        curve = np.clip(curve, 0, 1)
        curve = np.maximum.accumulate(curve)

        # 💰 Final cumulative paid
        ultimate = np.random.uniform(5000, 50000)
        cumulative_paid = np.round(curve * ultimate).astype(float)
        cumulative_paid[-1] = ultimate

        # 📦 Build row
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
    print("💣 Saved data/all_contracts.csv with **extreme** disruption.")
    return df
