import os
import mlflow
import dotenv

dotenv.load_dotenv()
mlflow.set_tracking_uri("http://127.0.0.1:5001")