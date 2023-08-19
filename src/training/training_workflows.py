import os
from pathlib import Path
from typing import Literal, Optional

import click
from prefect import flow

import preprocessing.utils as utils
from training.trainer import Trainer


@flow(name="load_data_for_trainer")
def load_data_for_trainer(
    load_from_s3: bool,
    features_folder: str,
    target_encoder_file: str,
    feature_names_file: str,
    train_file_name: str,
    val_file_name: str,
    test_file_name: str,
) -> tuple:
    if load_from_s3:
        data = utils.get_data_from_s3(
            features_folder, bucket_type="features", read_binaries=True
        )
        target_encoder = data[target_encoder_file]
        feature_names = data[feature_names_file]
        train_data = data[train_file_name]
        val_data = data[val_file_name]
        test_data = data[test_file_name]
    else:
        target_encoder = utils.load_obj(Path(features_folder) / target_encoder_file)
        feature_names = utils.load_obj(Path(features_folder) / feature_names_file)
        train_data = utils.load_obj(Path(features_folder) / train_file_name)
        val_data = utils.load_obj(Path(features_folder) / val_file_name)
        test_data = utils.load_obj(Path(features_folder) / test_file_name)
    return target_encoder, feature_names, train_data, val_data, test_data


@flow(name="training_runner")
def training_runner(
    tracking_uri: str,
    clf_training_func_name: Literal[
        "run_xgboost_training",
        "run_random_forest_training",
        "run_kneighbors_classifier_training",
    ],
    features_folder: str,
    load_from_s3: bool,
    target_encoder_file: str,
    feature_names_file: str,
    train_file_name: str,
    val_file_name: Optional[str] = None,
    test_file_name: Optional[str] = None,
    metric_name: str = "sklearn.metrics:accuracy_score",
    classifier_kwargs_file: Optional[str] = None,
    target_column_name: str = "genre",
    random_state: int = 42,
    n_trials_for_hyperparams: int = 20,
):
    (
        target_encoder,
        feature_names,
        train_data,
        val_data,
        test_data,
    ) = load_data_for_trainer(
        load_from_s3,
        features_folder,
        target_encoder_file,
        feature_names_file,
        train_file_name,
        val_file_name,
        test_file_name,
    )
    classfier_kwargs = (
        utils.load_config_from_yaml(classifier_kwargs_file)
        if classifier_kwargs_file is not None
        else None
    )

    trainer = Trainer(
        tracking_uri,
        clf_training_func_name,
        train_data,
        target_encoder,
        feature_names,
        metric_name,
        val_data,
        test_data,
        classfier_kwargs,
        target_column_name,
        random_state,
        n_trials_for_hyperparams,
    )
    trainer.run()


@click.command()
@click.option("--tracking_uri", type=str)
@click.option(
    "--clf_training_func_name",
    type=click.Choice(
        [
            "run_xgboost_training",
            "run_random_forest_training",
            "run_kneighbors_classifier_training",
        ]
    ),
)
@click.option("--features_folder", type=str)
@click.option("--load_from_s3", is_flag=True)
@click.option("--target_encoder_file", type=str)
@click.option("--feature_names_file", type=str)
@click.option("--train_file_name", type=str)
@click.option("--val_file_name", type=str)
@click.option("--test_file_name", type=str)
@click.option("--metric_name", type=str)
@click.option("--classifier_kwargs_file", type=str)
@click.option("--target_column_name", type=str, default="genre")
@click.option("--random_state", type=int, default=42)
@click.option("--n_trials_for_hyperparams", type=int, default=20)
@click.option("--deployments_folder", type=str, default="deployments")
def main_training_wrapper(
    tracking_uri: str,
    clf_training_func_name: Literal[
        "run_xgboost_training",
        "run_random_forest_training",
        "run_kneighbors_classifier_training",
    ],
    features_folder: str,
    load_from_s3: bool,
    target_encoder_file: str,
    feature_names_file: str,
    train_file_name: str,
    val_file_name: Optional[str] = None,
    test_file_name: Optional[str] = None,
    metric_name: str = "sklearn.metrics:accuracy_score",
    classifier_kwargs_file: Optional[str] = None,
    target_column_name: str = "genre",
    random_state: int = 42,
    n_trials_for_hyperparams: int = 20,
    deployments_folder: str = "deployments",
):
    CUR_PATH = Path(__file__).parent.resolve()
    FILE_NAME = Path(__file__).name
    Path(CUR_PATH).joinpath(deployments_folder).mkdir(exist_ok=True, parents=True)

    deployment_kwargs = {
        "name": clf_training_func_name,
        "entrypoint": f"{CUR_PATH}/{FILE_NAME}:training_runner",
        "work_pool_name": os.getenv("PREFECT_POOL"),
        "flow_name": "training_runner",
        "parameters": {
            "tracking_uri": tracking_uri,
            "clf_training_func_name": clf_training_func_name,
            "features_folder": features_folder,
            "load_from_s3": load_from_s3,
            "target_encoder_file": target_encoder_file,
            "feature_names_file": feature_names_file,
            "train_file_name": train_file_name,
            "val_file_name": val_file_name,
            "test_file_name": test_file_name,
            "metric_name": metric_name,
            "classifier_kwargs_file": classifier_kwargs_file,
            "target_column_name": target_column_name,
            "random_state": random_state,
            "n_trials_for_hyperparams": n_trials_for_hyperparams,
        },
        "path": f"{CUR_PATH}/{deployments_folder}",
    }
    logger = utils.init_logger(__file__, "INFO")
    utils.create_and_run_deployment(deployment_kwargs, logger)


if __name__ == "__main__":
    main_training_wrapper()
