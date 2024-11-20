import os
import mlflow
import dotenv

dotenv.load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
mlflow.set_tracking_uri("http://127.0.0.1:5001")