import os
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
import click
import numpy as np
import pandas as pd
import preprocessing.utils as utils
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner
from preprocessing.audio import Audio
from preprocessing.feature_extractor import FeatureExtractor
from sklearn.preprocessing import LabelEncoder


@flow(name="feature_dicts_to_df", validate_parameters=False)
def feature_dicts_to_df(
    feats: List[Dict[str, Union[str, Number]]], target_column: Optional[str] = None
):
    df = pd.DataFrame(feats, columns=feats[0].keys())
    if target_column is not None:
        encoder = utils.label_encoder_from_labels(df[target_column].values)
    else:
        encoder = None
    return df, encoder


@flow(name="split_input_by_input_and_target", validate_parameters=False)
def input_target_from_df(
    df: pd.DataFrame,
    target_column: str,
    encoder: Optional[LabelEncoder] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df_without_target = df.drop(target_column, axis=1)
    X = df_without_target.values
    y = df[target_column].values
    if encoder is not None:
        y = encoder.transform(y)
    return (X, y, list(df_without_target.columns))


@flow(name="train_val_split", validate_parameters=False)
def train_val_split_from_df(
    df: pd.DataFrame,
    target_column: str,
    encoder: LabelEncoder,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
    X, y, column_names = input_target_from_df(df, target_column)
    y_ = encoder.transform(y)
    train_ids, val_ids = utils.get_train_valid_ids(X, y_, val_size, seed)

    train_data = X[train_ids], y[train_ids]
    val_data = X[val_ids], y[val_ids]

    return train_data, val_data, column_names


@task(name="feature_extraction_from_file", log_prints=True, tags=["preprocessing"])
def run_feature_extraction_from_audio_file(
    file: str,
    to_pandas: bool = False,
    feature_extractor: Optional[FeatureExtractor] = None,
) -> Union[List[Dict[str, Union[str, Number]]], pd.DataFrame, Any]:
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    audio = Audio(file, sr=None, begin_offset=0, duration=30)
    # failed to load audio, return nothing
    if audio.audio is None:
        return []
    feature_dict = feature_extractor(audio)
    print(f"Features extracted from file {file}")
    if to_pandas:
        res = pd.DataFrame(data=[feature_dict], columns=feature_dict.keys())
        return res
    return [feature_dict]


@task(name="get_files_list", tags=["preprocessing"], persist_result=True)
def get_files_list(folder: str) -> List[str]:
    return [
        os.path.join(dir, file) for dir, _, files in os.walk(folder) for file in files
    ]


@flow(
    name="feature_extraction_all",
    description="Extracts features from all audio files given folder",
    validate_parameters=False,
    task_runner=DaskTaskRunner(),
    persist_result=True,
)
def run_feature_extraction_all(
    folder: str,
    val_split_prop: Optional[float] = None,
    target_column_name: str = "genre",
):
    all_files = get_files_list(folder)
    feature_extractor = FeatureExtractor()

    prefect_futures = []
    for file in all_files:
        prefect_futures.append(
            run_feature_extraction_from_audio_file.submit(
                file, False, feature_extractor
            )
        )

    feature_dicts = []
    for prefect_future in prefect_futures:
        feature_dicts.extend(prefect_future.result())

    feature_dicts, encoder = feature_dicts_to_df(
        feature_dicts, target_column=target_column_name
    )

    if val_split_prop is not None:
        train_data, val_data, column_names = train_val_split_from_df(
            feature_dicts, target_column_name, encoder, val_split_prop
        )
        return train_data, val_data, column_names, encoder

    X, y, column_names = input_target_from_df(
        feature_dicts,
        target_column_name,
    )

    return ((X, y), (), column_names, encoder)


@click.command()
@click.option("--input_folder", type=str)
@click.option("--output_folder", type=str, default="/data/processed")
@click.option("--val_split_prop", type=float)
@click.option("--mode", type=click.Choice(["train", "test"]))
def main_preprocessing(
    input_folder: str,
    output_folder: str,
    mode: str,
    val_split_prop: Optional[float] = None,
):
    train_data, val_data, column_names, encoder = run_feature_extraction_all(
        input_folder, val_split_prop
    )
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    utils.save_obj(train_data, f"{output_folder}/{mode}.pkl")
    utils.save_obj(encoder, f"{output_folder}/target_encoder.pkl")
    utils.save_obj(column_names, f"{output_folder}/feature_names.pkl")

    session = boto3.session.Session()
    s3_client = session.client(
        service_name="s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    )
    s3_client.put_object(
        Body=f"{output_folder}/{mode}.pkl",
        Bucket="audioprocessor",
        Key=f"{mode}.pkl",
    )
    s3_client.put_object(
        Body=f"{output_folder}/target_encoder.pkl",
        Bucket="audioprocessor",
        Key=f"target_encoder.pkl",
    )
    s3_client.put_object(
        Body=f"{output_folder}/feature_names.pkl",
        Bucket="audioprocessor",
        Key=f"feature_names.pkl",
    )

    if val_split_prop is not None and len(val_data):
        utils.save_obj(val_data, f"{output_folder}/val.pkl")
        s3_client.put_object(
            Body=f"{output_folder}/val.pkl",
            Bucket="audioprocessor",
            Key=f"val.pkl",
        )


if __name__ == "__main__":
    main_preprocessing()
