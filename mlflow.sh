mlflow server \
    --backend-store-uri sqlite:///mlflow_data/mlflow.db \
    --default-artifact-root s3://monosemanticity-mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5001