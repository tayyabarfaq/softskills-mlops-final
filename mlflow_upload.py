import os
import mlflow
import dagshub

os.environ["MLFLOW_TRACKING_USERNAME"] = "tayyabarfaq"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1ad859eaa3a688b39a56b84cb949867dd3db53d"

dagshub.init(
    repo_owner="tayyabarfaq",
    repo_name="softskills-mlops-final",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/tayyabarfaq/softskills-mlops-final.mlflow")

# IMPORTANT: Experiment MUST already exist in DAGsHub
mlflow.set_experiment("softskills_experiment")

with mlflow.start_run(run_name="softskills_model_upload"):

    mlflow.log_artifact("softskills_model.pt")
    mlflow.log_artifact("softskills_model")

print("Model + Tokenizer uploaded successfully!")
