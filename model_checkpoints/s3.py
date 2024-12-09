import boto3
import pandas as pd
import io
import joblib

# Initialize S3 Resource
s3 = boto3.resource(
    service_name="s3",
    region_name="us-east-2"
)

def load_data_from_s3(bucket, file_path ,  **kwargs):
    s3_object = s3.Object(bucket, file_path)
    s3_object.load()
    s3_response = s3_object.get()
    s3_data = s3_response['Body'].read()
    
    if file_path.endswith('.csv'):
        return pd.read_csv(io.BytesIO(s3_data),  **kwargs)  # Load as DataFrame if CSV
    elif file_path.endswith('.pkl'):
        return joblib.load(io.BytesIO(s3_data))   
    else:
        print(f"Unsupported file type for file {file_path}. Please use .csv or .pkl files.")
        return None
