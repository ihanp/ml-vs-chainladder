import pandas as pd
import numpy as np
import os

def create_residual_training_data_varying(df, max_dev=9):
    inputs = []
    targets = []

    for _, row in df.iterrows():
        full_curve = [row[f"dev_{i}"] for i in range(max_dev + 1)]
        for observed_dev in range(1, max_dev):  # dev_1 to dev_8
            known = full_curve[:observed_dev]
            padded_input = known + [0] * (max_dev - observed_dev)
            known_paid = known[-1]
            target_residual = row["ultimate"] - known_paid

            inputs.append(padded_input)
            targets.append(target_residual)

    input_df = pd.DataFrame(inputs, columns=[f"dev_{i}" for i in range(max_dev)])
    target_series = pd.Series(targets, name="residual_to_ultimate")
    return input_df, target_series


def prepare_train_test_split():
    os.makedirs("data", exist_ok=True)

    # Load full synthetic data
    df = pd.read_csv("data/all_contracts.csv")

    # Split into train and test based on policy year
    train_df = df[df["policy_year"] <= 2014].reset_index(drop=True)
    test_df = df[df["policy_year"] > 2014].reset_index(drop=True)

    # Save to CSV
    train_df.to_csv("data/train_contracts.csv", index=False)
    test_df.to_csv("data/test_contracts.csv", index=False)

    print(f"Saved {len(train_df)} training contracts and {len(test_df)} test contracts.")

    # Create training pairs
    X_train, y_train = create_residual_training_data_varying(train_df, max_dev=9)

    X_train.to_csv("data/X_train.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)

    print("Training pairs generated (Residual-to-Ultimate):")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    return X_train, y_train, test_df
