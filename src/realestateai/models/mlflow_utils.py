import os

import mlflow
import pandas as pd


def log_pd_dataframe(df: pd.DataFrame, artifact_path: str, file_format="csv"):
    # assuming that you are in the mlflow.start_run()
    if file_format == "csv":
        df.to_csv(artifact_path)
    elif file_format == "parquet":
        df.to_parquet(artifact_path)

    mlflow.log_artifact(artifact_path, artifact_path)
    os.remove(artifact_path)
