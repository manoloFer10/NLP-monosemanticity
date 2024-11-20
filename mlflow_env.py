import os
import mlflow
import sys

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
mlflow.set_tracking_uri("http://34.176.189.11:5000/")