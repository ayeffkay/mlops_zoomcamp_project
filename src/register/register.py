import ast
import glob
import os
from pathlib import Path
from typing import List, Literal, Optional

import click
import mlflow
import preprocessing.utils as utils
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task

mlflow.set_tracking_uri(os.getenv("BACKEND_STORE_URI"))


@task(name="prepare_for_deploy", log_prints=True)
def prepare_for_deploy(
    registered_model_name: str = "best_model",
    n_models: int = 1,
    target_model_dir: str = "deploy/triton_models/predictor",
    target_encoder_dir: str = "deploy/triton_models/post_processor/1",
    model_file: str = "model.*",
    target_encoder_file: str = "target_encoder.pkl",
    artifact_model_folder: str = "model",
):
    client = MlflowClient(tracking_uri=os.getenv("BACKEND_STORE_URI"))
    run_ids = [
        version.run_id
        for version in client.search_registered_models(
            f"name='{registered_model_name}'"
        )[0].latest_versions[-n_models:]
    ]
    for i, run_id in enumerate(run_ids, 1):
        Path(f"/data/artifacts/{i}").mkdir(parents=True, exist_ok=True)
        model_artifacts_path = client.download_artifacts(
            run_id=run_id,
            path=artifact_model_folder,
            dst_path=f"/data/artifacts/{i}",
        )
        model_file = list(glob.glob(f"{Path(model_artifacts_path)/ model_file}"))[0]
        target_artifacts_path = client.download_artifacts(
            run_id=run_id,
            path=artifact_model_folder,
            dst_path=f"/data/artifacts/{i}",
        )
        target_encoder_path = str(Path(target_artifacts_path) / target_encoder_file)

        dst_model_path = os.path.join(
            target_model_dir, str(i), os.path.basename(model_file)
        )
        dst_encoder_path = os.path.join(target_encoder_dir, target_encoder_file)
        if os.path.exists(dst_model_path):
            os.remove(dst_model_path)
        if os.path.exists(dst_encoder_path):
            os.remove(dst_encoder_path)
        Path(os.path.dirname(dst_model_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(dst_encoder_path)).mkdir(parents=True, exist_ok=True)
        os.popen(f"cp {model_file} {dst_model_path}")
        os.popen(f"cp {target_encoder_path} {dst_encoder_path}")


@task(name="run_id_from_registered_model")
def get_run_id_from_registered_model(
    client: mlflow.tracking.client.MlflowClient, model_name: str = "best_model"
) -> str:
    run_id = (
        client.search_registered_models(f"name='{model_name}'")[0]
        .latest_versions[0]
        .run_id
    )
    return run_id


@flow(name="export_to_onnx", validate_parameters=False, log_prints=True)
def export_to_onnx(
    client: Optional[mlflow.tracking.client.MlflowClient] = None,
    run_id: Optional[str] = None,
):
    """
    Fails for KNearestNeighbors with some distance metrics (e.g. `canberra`).
    Sklearn models can't be loaded inside TritonInferenceServer with onnxruntime backend
    So I rejected to use this flow, it works with xgboost only.
    """
    if not client:
        client = MlflowClient(tracking_uri=os.getenv("BACKEND_STORE_URI"))
    if not run_id:
        run_id = get_run_id_from_registered_model(client)

    run = client.get_run(run_id)
    features_shape = int(
        ast.literal_eval(run.inputs.dataset_inputs[0].dataset.profile)["features_size"]
    )
    # for debugging
    local_dst_path = "/data/artifacts/run_id"
    Path(local_dst_path).mkdir(parents=True, exist_ok=True)
    model_artifacts_path = client.download_artifacts(
        run_id=run_id, path="model", dst_path=local_dst_path
    )
    model_file = list(glob.glob(f"{Path(model_artifacts_path)/'model.*'}"))[0]
    onnx_model = None

    if os.path.splitext(model_file)[-1] == ".xgb":
        import xgboost as xgb
        from onnxconverter_common.data_types import FloatTensorType
        from onnxmltools.convert import convert_xgboost

        model = xgb.XGBClassifier()
        model.load_model(fname=model_file)
        initial_type = [("FEATURES", FloatTensorType([None, features_shape]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
    else:
        import numpy as np
        from skl2onnx import to_onnx

        model = utils.load_obj(model_file)
        options = {id(model): {"zipmap": False}}
        dummy_inputs = np.random.randn(1, features_shape)
        onnx_model = to_onnx(model, dummy_inputs, options=options)
        onnx_model_file = Path(local_dst_path) / "model.onnx"
        with open(onnx_model_file, "wb") as f:
            f.write(onnx_model.SerializeToString())

        client.log_artifact(
            run_id=run_id, local_path=onnx_model_file, artifact_path="model"
        )


@task(name="get_best_run_id", log_prints=True)
def get_best_run_id(
    client: mlflow.tracking.client.MlflowClient,
    experiment_ids: List[str],
    metric_name: Literal["test_geometric_mean_score", "val_geometric_mean_score"],
    n_models: int = 1,
) -> tuple:
    best_run = client.search_runs(
        experiment_ids=experiment_ids,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=n_models,
        order_by=[f"metrics.{metric_name} DESC"],
        filter_string='attribute.run_name LIKE "best_%"',
    )
    info = [(run.info.run_id, run.data.metrics[metric_name]) for run in best_run]
    ids, metrics = list(zip(*info))
    return ids, metrics


@flow(name="translit_best_registered_to_stage", log_prints=True)
def transit_best_registered_to_stage(
    tracking_uri: str,
    by_name: str,
    stage: Literal["Staging", "Production", "Archived"] = "Production",
):
    client = MlflowClient(tracking_uri=tracking_uri)
    version = int(
        client.search_registered_models(f"name='{by_name}'")[0]
        .latest_versions[0]
        .version
    )
    client.transition_model_version_stage(name=by_name, version=version, stage=stage)


@flow(name="register_best_model", log_prints=True)
def register_best_model(
    tracking_uri: str,
    registered_model_name: str,
    metric_name: str,
    name_pattern: str = "run_%",
    transit_to_stage: Optional[Literal["Staging", "Production", "Archived"]] = None,
    n_models: int = 1,
):
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = [
        experiment.experiment_id
        for experiment in client.search_experiments(
            filter_string=f"attribute.name LIKE '{name_pattern}'"
        )
    ]

    best_run_ids, _ = get_best_run_id(client, experiment_ids, metric_name, n_models)
    for best_run_id in best_run_ids[::-1]:
        best_model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri=best_model_uri, name=registered_model_name)
        # fails for some models, I had to comment this
        # export_to_onnx(client, best_run_id)
        if transit_to_stage:
            transit_best_registered_to_stage(
                tracking_uri, registered_model_name, transit_to_stage
            )
            if transit_to_stage == "Production":
                # we will use two models in production to switch between them when data drift occurs
                prepare_for_deploy(registered_model_name, n_models=2)


@click.command()
@click.option("--tracking_uri", type=str)
@click.option("--registered_model_name", type=str)
@click.option(
    "--select_by_metric_name",
    type=click.Choice(["test_geometric_mean_score", "val_geometric_mean_score"]),
)
@click.option(
    "--transit_to_stage",
    type=click.Choice(["Staging", "Production", "Archived"]),
)
@click.option("--name_pattern", type=str, default="run_%")
@click.option("--n_models", type=int, default=1)
@click.option("--deployments_folder", type=str, default="deployments")
def register_wrapper(
    tracking_uri: str,
    registered_model_name: str,
    select_by_metric_name: Literal[
        "test_geometric_mean_score", "val_geometric_mean_score"
    ],
    transit_to_stage: Optional[Literal["Staging", "Production", "Archived"]] = None,
    name_pattern: str = "run_%",
    n_models: int = 1,
    deployments_folder: str = "deployments",
):
    CUR_PATH = Path(__file__).parent.resolve()
    FILE_NAME = Path(__file__).name
    Path(CUR_PATH).joinpath(deployments_folder).mkdir(exist_ok=True, parents=True)
    mlflow.set_tracking_uri(os.getenv("BACKEND_STORE_URI"))

    deployment_kwargs = {
        "name": "register_best",
        "entrypoint": f"{CUR_PATH}/{FILE_NAME}:register_best_model",
        "work_pool_name": os.getenv("PREFECT_POOL"),
        "flow_name": "register_best_model",
        "parameters": {
            "tracking_uri": tracking_uri,
            "registered_model_name": registered_model_name,
            "metric_name": select_by_metric_name,
            "transit_to_stage": transit_to_stage,
            "name_pattern": name_pattern,
            "n_models": n_models,
        },
        "path": f"{CUR_PATH}/{deployments_folder}",
    }
    logger = utils.init_logger(__file__, "INFO")
    utils.create_and_run_deployment(deployment_kwargs, logger)


if __name__ == "__main__":
    register_wrapper()
