import os
import mlflow
import sys

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
mlflow.set_tracking_uri("http://127.0.0.1:5001")