import os
from pathlib import Path
from typing import Literal

import click
import preprocessing.utils as utils


@click.command()
@click.option("--folder", type=str)
@click.option("--deployments_folder", type=str)
@click.option(
    "--mode",
    type=click.Choice(["train", "train_subset", "test", "features", "artifacts"]),
)
@click.option(
    "--bucket_type",
    type=click.Choice(["raw", "features", "mlflow"]),
)
def create_put_data_deployment(
    folder: str,
    deployments_folder: str,
    mode: str,
    bucket_type: Literal["raw", "features", "mlflow"],
):
    CUR_PATH = Path(__file__).parent.resolve()

    Path(CUR_PATH).joinpath(deployments_folder).mkdir(exist_ok=True, parents=True)
    deployment_kwargs = {
        "name": f"put_{mode}_{bucket_type}_to_s3",
        "entrypoint": f"{CUR_PATH}/utils.py:put_data_to_s3",
        "work_pool_name": os.getenv("PREFECT_POOL"),
        "flow_name": "put_data_to_s3",
        "parameters": {"folder": folder, "bucket_type": bucket_type},
        "path": f"{CUR_PATH}/{deployments_folder}",
    }
    logger = utils.init_logger(__file__, "INFO")
    utils.create_and_run_deployment(deployment_kwargs, logger)


if __name__ == "__main__":
    create_put_data_deployment()
