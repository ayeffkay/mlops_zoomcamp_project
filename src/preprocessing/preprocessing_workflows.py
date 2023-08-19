import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import click
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
from prefect import flow
from prefect_dask.task_runners import DaskTaskRunner
from sklearn.preprocessing import LabelEncoder

import preprocessing.utils as utils
from preprocessing.feature_extractor import (
    FeatureExtractor,
    run_feature_extraction_from_audio_file,
)


@flow(name="feature_dicts_to_df", validate_parameters=False)
def feature_dicts_to_df(
    feats: Sequence[Any], target_column: Optional[str] = None
) -> Tuple[pd.core.frame.DataFrame, Any]:
    df = pd.DataFrame(feats, columns=feats[0].keys())
    if target_column is not None:
        encoder = utils.label_encoder_from_labels(df[target_column].values)
    else:
        encoder = None
    return df, encoder


@flow(name="split_input_by_input_and_target", validate_parameters=False)
def input_target_from_df(
    df: pd.core.frame.DataFrame,
    target_column: str,
    encoder: Optional[LabelEncoder] = None,
) -> Tuple[np.ndarray, Union[ExtensionArray, np.ndarray], List[str]]:
    df_without_target = df.drop(target_column, axis=1)
    X = df_without_target.values
    y = df[target_column].values
    if encoder is not None:
        y = encoder.transform(y)
    return X, y, list(df_without_target.columns)


@flow(name="train_val_split", validate_parameters=False)
def train_val_split_from_df(
    df: pd.core.frame.DataFrame,
    target_column: str,
    encoder: LabelEncoder,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    X: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    column_names: List[str] = []
    X, y, column_names = input_target_from_df(df, target_column)
    y_ = encoder.transform(y)
    train_ids, val_ids = utils.get_train_valid_ids(X, y_, val_size, seed)

    train_data = [X[train_ids], y[train_ids]]
    val_data = [X[val_ids], y[val_ids]]

    return train_data, val_data, column_names


@flow(
    name="feature_extraction_all",
    description="Extracts features from all audio files given folder",
    validate_parameters=False,
    task_runner=DaskTaskRunner(),
    persist_result=True,
)
def run_feature_extraction_all(
    all_files: List[str],
    audio_config_file: str,
    val_split_prop: Optional[float] = None,
    target_column_name: str = "genre",
) -> Tuple[List[np.ndarray], List[Any], List[str], Any]:
    feature_extractor = FeatureExtractor()

    prefect_futures = []
    for file in all_files:
        prefect_futures.append(
            run_feature_extraction_from_audio_file.submit(
                file, audio_config_file, False, feature_extractor
            )
        )

    feature_dicts: List[dict] = []
    for prefect_future in prefect_futures:
        feature_dicts.extend(prefect_future.result())

    feature_df: pd.DataFrame = pd.DataFrame()
    encoder: LabelEncoder = LabelEncoder()
    feature_df, encoder = feature_dicts_to_df(
        feature_dicts, target_column=target_column_name
    )

    if val_split_prop:
        train_data: List[np.ndarray] = []
        val_data: List[np.ndarray] = []
        column_names: List[str] = []
        train_data, val_data, column_names = train_val_split_from_df(
            feature_df, target_column_name, encoder, val_split_prop
        )
        return train_data, val_data, column_names, encoder

    X: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    X, y, column_names = input_target_from_df(
        feature_df,
        target_column_name,
    )

    return [X, y], [], column_names, encoder


@flow(name="main_preprocessing", log_prints=True)
def main_preprocessing(
    input_folder: str,
    audio_config_file: str,
    output_folder: str,
    mode: str,
    val_split_prop: Optional[float] = None,
    load_from_s3: bool = False,
    put_outputs_to_s3: bool = False,
):
    train_data: List[np.ndarray] = []
    val_data: List[np.ndarray] = []
    column_names: List[np.ndarray] = []
    encoder: LabelEncoder = LabelEncoder()

    if load_from_s3:
        all_files = utils.get_data_from_s3(input_folder, "raw")
    else:
        all_files = utils.get_files_list(input_folder)

    print(f"Starting to process {len(all_files)} files...")
    train_data, val_data, column_names, encoder = run_feature_extraction_all(
        all_files, audio_config_file, val_split_prop
    )
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    utils.save_obj(train_data, f"{output_folder}/{mode}.pkl")
    utils.save_obj(encoder, f"{output_folder}/target_encoder.pkl")
    utils.save_obj(column_names, f"{output_folder}/feature_names.pkl")
    if val_split_prop is not None and len(val_data):
        utils.save_obj(val_data, f"{output_folder}/val.pkl")

    # prefect doesn't recognize s3 from localstack,
    # so prefect.filesystems.S3 doesn't work here :(
    if put_outputs_to_s3:
        utils.put_data_to_s3(output_folder, "features")


@click.command()
@click.option("--input_folder", type=str)
@click.option("--audio_config_file", type=str)
@click.option("--output_folder", type=str, default="/data/processed")
@click.option("--mode", type=click.Choice(["train", "train_subset", "test"]))
@click.option("--run_as_deployment", is_flag=True)
@click.option("--load_from_s3", is_flag=True)
@click.option("--put_outputs_to_s3", is_flag=True)
@click.option("--val_split_prop", type=float, default=0)
@click.option("--deployments_folder", type=str, default="deployments")
def main_preprocessing_runner(
    input_folder: str,
    audio_config_file: str,
    output_folder: str,
    mode: str,
    run_as_deployment: bool,
    load_from_s3: bool,
    put_outputs_to_s3: bool,
    val_split_prop: Optional[float] = None,
    deployments_folder: str = "deployments",
):
    CUR_PATH = Path(__file__).parent.resolve()
    FILE_NAME = Path(__file__).name

    preproc_kwargs = {
        "input_folder": input_folder,
        "audio_config_file": f"{CUR_PATH}/{audio_config_file}",
        "output_folder": output_folder,
        "val_split_prop": val_split_prop,
        "mode": mode,
        "load_from_s3": load_from_s3,
        "put_outputs_to_s3": put_outputs_to_s3,
    }
    Path(CUR_PATH).joinpath(deployments_folder).mkdir(parents=True, exist_ok=True)
    if run_as_deployment:
        deployment_kwargs = {
            "name": f"{mode}_data",
            "entrypoint": f"{CUR_PATH}/{FILE_NAME}:main_preprocessing",
            "work_pool_name": os.getenv("PREFECT_POOL"),
            "flow_name": "main_preprocessing",
            "parameters": preproc_kwargs,
            "path": f"{CUR_PATH}/{deployments_folder}",
        }
        logger = utils.init_logger(__file__, "INFO")
        utils.create_and_run_deployment(deployment_kwargs, logger)
    else:
        main_preprocessing(**preproc_kwargs)


if __name__ == "__main__":
    main_preprocessing_runner()
