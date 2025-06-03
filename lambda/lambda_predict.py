import json
import boto3
import joblib
import numpy as np
import pandas as pd
import io

# Load model and scalers once, outside handler for efficiency
s3 = boto3.client("s3")
bucket = "ml-chainladder-data"

def load_pickle_from_s3(key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(io.BytesIO(obj['Body'].read()))

mlp = load_pickle_from_s3("mlp_model.pkl")
input_scaler = load_pickle_from_s3("input_scaler.pkl")
target_scaler = load_pickle_from_s3("target_scaler.pkl")

def lambda_handler(event, context):
    try:
        # Parse JSON input
        input_data = json.loads(event["body"]) if "body" in event else event
        dev_values = input_data.get("dev", [])

        if not dev_values or not isinstance(dev_values, list):
            return {"statusCode": 400, "body": json.dumps({"error": "Missing or invalid 'dev' field"})}

        # Pad dev values to max_dev = 9
        max_dev = 9
        cumulative = dev_values[:max_dev]
        padded_input = cumulative + [0.0] * (max_dev - len(cumulative))
        known_paid = cumulative[-1] if cumulative else 0.0

        # Convert to DataFrame and scale
        X = pd.DataFrame([padded_input], columns=[f"dev_{i}" for i in range(max_dev)])
        X_scaled = input_scaler.transform(X)

        # Predict residual
        residual_scaled = mlp.predict(X_scaled)
        residual = target_scaler.inverse_transform(residual_scaled.reshape(-1, 1)).flatten()[0]

        # Calculate predicted ultimate
        predicted_ultimate = known_paid + residual

        return {
            "statusCode": 200,
            "body": json.dumps({
                "predicted_ultimate": round(predicted_ultimate, 2),
                "residual": round(residual, 2),
                "known_paid": round(known_paid, 2)
            })
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
