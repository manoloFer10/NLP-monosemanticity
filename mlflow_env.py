import os
import mlflow
import dotenv

dotenv.load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
mlflow.set_tracking_uri("http://34.176.189.11:5000/")