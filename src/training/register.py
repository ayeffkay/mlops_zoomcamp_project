from typing import List

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow
from preprocessing.utils import get_current_time


@flow(name="register_best_model")
def register_best_model(
    tracking_uri: str,
    experiment_names: List[str],
    registered_model_name: str,
    metric_name: str,
):
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = [
        client.get_experiment_by_name(experiment_name).experiment_id
        for experiment_name in experiment_names
    ]

    best_run = client.search_runs(
        experiment_ids=experiment_ids,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=[f"metrics.{metric_name} DESC"],
        filter_string='attribute.run_name LIKE "best_%"',
    )[0]

    best_run_id = best_run.info.run_id
    best_model_uri = f"runs:/{best_run_id}/model"

    registered_model_name = f"{registered_model_name}_{get_current_time()}"
    mlflow.register_model(model_uri=best_model_uri, name=registered_model_name)
