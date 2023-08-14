import calendar
import datetime
import logging
import os
import pickle
import pprint
import sys
import time
from logging import Formatter, StreamHandler
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import boto3
import numpy as np
import pandas as pd
import ruamel.yaml
from prefect import flow, task
from prefect.deployments import Deployment, run_deployment
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


def create_and_run_deployment(
    deployment_kwargs: Dict[str, Any], logger: Optional[logging.Logger] = None
):
    if logger:
        logger.info(
            f"Created deployment with the following arguments: {pprint.pprint(deployment_kwargs)}"
        )
    deployment = Deployment(**deployment_kwargs)
    _ = deployment.apply()
    if not os.path.exists(deployment.path):
        os.makedirs(deployment.path)

    save_path = str(
        os.path.join(deployment.path, f"{deployment.flow_name}-{deployment.name}.yaml")
    )
    deployment.to_yaml(save_path)
    if logger:
        logger.info(f"Deployment config saved into {save_path}.")

    deployment_name = f"{deployment.flow_name}/{deployment.name}"
    if logger:
        logger.info(f"Started running {deployment.name} deployment...")
    response = run_deployment(deployment_name)
    if logger:
        logger.info(f"Finished running {deployment.name} deployment!")
    return response


def get_current_timestamp() -> int:
    current_gmt = time.gmtime()
    time_stamp = calendar.timegm(current_gmt)
    return time_stamp


def concat_features_with_target(np_data: List[np.ndarray]) -> np.ndarray:
    return np.hstack((np_data[0], np_data[1].reshape(-1, 1)))


def df_from_numpy(
    np_data: List[np.ndarray],
    features_names: List[str],
    target_name: str,
) -> pd.DataFrame:
    np_data_concat = concat_features_with_target(np_data)
    columns = features_names + [target_name]
    return pd.DataFrame(np_data_concat, columns=columns)


def init_logger(app_name: str, logging_level: str) -> logging.Logger:
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, logging_level))

    handler = StreamHandler(stream=sys.stdout)
    handler.setFormatter(Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def decode_str_vfunc():
    return np.vectorize(lambda x: x.decode("utf-8"))


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")


def str2int_or_float(numeric_str: str) -> Union[int, float]:
    if numeric_str.isdigit():
        return int(numeric_str)
    return float(numeric_str)


def str2num_from_params(
    params_dict: Dict[str, str]
) -> Dict[str, Union[str, int, float]]:
    return {
        name: (str2int_or_float(value) if value.isnumeric() else value)
        for name, value in params_dict.items()
    }


def load_obj(path: str) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_obj(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_config_from_yaml(file: str) -> Dict[str, Any]:
    yaml = ruamel.yaml.YAML()
    with open(file) as f:
        params = yaml.load(f)
    return params


def get_files_by_names_from_s3(
    bucket_name: str,
    filenames: str,
    save_to: Optional[str] = None,
) -> List[str]:
    s3_client = init_s3_client.fn()
    output_files = []
    for file in filenames:
        if save_to:
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            output_dir = Path(save_to) / os.path.basename(file)
            s3_client.download_file(bucket_name, file, output_dir)
            output_files.append(str(output_dir))
        elif not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
            s3_client.download_file(bucket_name, file, file)
            output_files.append(file)
    return output_files


def put_files_by_names_to_s3(bucket_name: str, filenames: str):
    s3_client = init_s3_client.fn()
    for file in filenames:
        s3_client.upload_file(
            Filename=file,
            Bucket=bucket_name,
            Key=file,
        )


@flow(name="put_data_to_s3")
def put_data_to_s3(folder: str, bucket_type: Literal["raw", "features", "mlflow"]):
    files = get_files_list(folder)
    s3_client = init_s3_client()
    bucket = get_s3_bucket_name(bucket_type)
    for file in files:
        s3_client.upload_file(
            Filename=file,
            Bucket=bucket,
            Key=file,
        )


@flow(name="get_data_from_s3")
def get_data_from_s3(
    prefix: str,
    bucket_type: Literal["raw", "features", "mlflow"],
    read_binaries: bool = False,
    save_to: Optional[str] = None,
) -> Union[List[str], Dict[str, List]]:
    s3_client = init_s3_client()
    bucket = get_s3_bucket_name(bucket_type)
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    s3_files = [obj["Key"] for page in pages for obj in page["Contents"]]
    output = {}
    for file in s3_files:
        if save_to and not os.path.exists(save_to):
            os.makedirs(save_to)
        elif not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        if bucket_type == "features" and read_binaries:
            binary_loaded = pickle.loads(
                s3_client.get_object(Bucket=bucket, Key=file)["Body"].read()
            )

            output.update({os.path.basename(file): binary_loaded})
        else:
            if save_to:
                s3_client.download_file(
                    bucket, file, Path(save_to) / os.path.basename(file)
                )
            else:
                s3_client.download_file(bucket, file, file)
    if len(output):
        return output
    return s3_files


@task(name="get_s3_bucket_name")
def get_s3_bucket_name(bucket_type: Literal["features", "mlflow", "raw"]) -> str:
    if bucket_type == "features":
        bucket = os.getenv("S3_FEATURES_BUCKET", "features")
    elif bucket_type == "mlflow":
        bucket = os.getenv("S3_MLFLOW_ARTIFACTS_BUCKET", "mlflow")
    else:
        bucket = os.getenv("S3_RAW_DATA_BUCKET", "raw")
    return bucket


@task(name="initialize_s3_client")
def init_s3_client() -> Any:
    session = boto3.session.Session()
    s3_client = session.client(
        service_name="s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    )
    return s3_client


@task(name="get_files_list", tags=["preprocessing"], persist_result=True)
def get_files_list(folder: str) -> List[str]:
    return [
        os.path.join(dir, file) for dir, _, files in os.walk(folder) for file in files
    ]


@task(name="encode_targets", tags=["preprocessing"], persist_result=True)
def label_encoder_from_labels(
    raw_labels: np.ndarray,
) -> LabelEncoder:
    encoder = LabelEncoder().fit(raw_labels)
    return encoder


def get_train_valid_ids(
    X: np.ndarray, y: np.ndarray, val_prop: float = 0.1, seed: int = 42
) -> List[np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_prop, random_state=seed)
    train_val_ids = list(sss.split(X, y))
    train_ids = train_val_ids[0][0]
    val_ids = train_val_ids[0][1]
    return [train_ids, val_ids]
