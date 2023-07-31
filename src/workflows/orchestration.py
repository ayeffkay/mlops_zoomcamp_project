from typing import Any, Dict

from prefect.deployments import Deployment, run_deployment


def create_and_run_deployment(deployment_kwargs: Dict[str, Any]):
    deployment = Deployment(**deployment_kwargs)
    deployment.apply()
    deployment_name = f"{deployment.flow_name}/{deployment.name}"
    response = run_deployment(deployment_name)
    return response
