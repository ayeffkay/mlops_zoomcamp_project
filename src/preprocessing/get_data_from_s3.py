import os
from pathlib import Path
from typing import Literal, Optional

import click

import preprocessing.utils as utils


@click.command()
@click.option("--folder", type=str)
@click.option("--deployments_folder", type=str)
@click.option("--bucket_type", type=click.Choice(["raw", "features", "mlflow"]))
@click.option("--save_to", type=str)
def create_get_data_deployment(
    folder: str,
    deployments_folder: str,
    bucket_type: Literal["raw", "features", "mlflow"],
    save_to: Optional[str] = None,
):
    CUR_PATH = Path(__file__).parent.resolve()

    Path(CUR_PATH).joinpath(deployments_folder).mkdir(exist_ok=True, parents=True)
    deployment_kwargs = {
        "name": f"get_{bucket_type}_from_s3",
        "entrypoint": f"{CUR_PATH}/utils.py:get_data_from_s3",
        "work_pool_name": os.getenv("PREFECT_POOL"),
        "flow_name": "get_data_from_s3",
        "parameters": {
            "prefix": folder,
            "bucket_type": bucket_type,
            "read_binaries": False,
            "save_to": save_to,
        },
        "path": f"{CUR_PATH}/{deployments_folder}",
    }
    logger = utils.init_logger(__file__, "INFO")
    utils.create_and_run_deployment(deployment_kwargs, logger)


if __name__ == "__main__":
    create_get_data_deployment()
