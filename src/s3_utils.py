import boto3
import pandas as pd
import joblib
import io

s3 = boto3.client("s3")
bucket = "ml-chainladder-data"

def load_csv_from_s3(key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        print(f"load_csv_from_s3 error: {e}")
        return None

def upload_csv_to_s3(df, key):
    with io.StringIO() as buffer:
        df.to_csv(buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

def load_model_from_s3(key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return joblib.load(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        print(f"load_model_from_s3 error: {e}")
        return None

def upload_model_to_s3(model, key):
    with io.BytesIO() as buffer:
        joblib.dump(model, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket, key)
