import asyncio
import os
import sys
from typing import List, Tuple

import black
import click
import numpy as np
import preprocessing.utils as utils
import tritonclient.grpc.aio as grpcclient
from prefect import flow, task

LOGGER = utils.init_logger(__name__, "INFO")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 16))


def get_triton_client(url: str, verbose: bool):
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
        )
    except Exception as e:
        LOGGER.error("channel creation failed: " + str(e))
        sys.exit()

    return triton_client


@task(name="generate_triton_inputs_and_outputs")
def get_inputs_and_outputs(file_names: List[str], input_name: str, output_name: str):
    inputs = [grpcclient.InferInput(input_name, [len(file_names), 1], "BYTES")]
    inputs_np = np.array(file_names, dtype=np.object_).reshape(-1, 1)
    inputs[0].set_data_from_numpy(inputs_np)
    outputs = [grpcclient.InferRequestedOutput(output_name)]
    return inputs, outputs


@task(name="decode_output")
def get_final_output(output_tensor, output_name: str):
    vfunc = utils.decode_str_vfunc()
    prediction = vfunc(output_tensor.as_numpy(output_name).flatten())
    return prediction


@flow(name="client_main_flow")
async def main(
    url: str,
    model_name: str,
    model_input_name: str,
    model_output_name: str,
    audio_files_names: str,
    verbose: bool,
) -> List[Tuple[str, str]]:
    triton_client = get_triton_client(url, verbose)
    audio_files_names_lst = audio_files_names.split(";")
    batches = [
        audio_files_names_lst[i : i + MAX_BATCH_SIZE]
        for i in range(0, len(audio_files_names_lst), MAX_BATCH_SIZE)
    ]
    all_predictions = []
    for batch in batches:
        inputs, outputs = get_inputs_and_outputs(
            batch, model_input_name, model_output_name
        )

        utils.put_files_by_names_to_s3(os.getenv("S3_CLIENT_RAW_BUCKET"), batch)
        LOGGER.info(
            f"Input files {batch} were saved to s3://{os.getenv('S3_CLIENT_RAW_BUCKET')}, requests were send to server..."
        )
        results = await triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        statistics = await triton_client.get_inference_statistics(model_name=model_name)
        LOGGER.info(statistics)
        if len(statistics.model_stats) != 1:
            LOGGER.error("FAILED: Inference Statistics")
            sys.exit(1)

        predictions = get_final_output(results, model_output_name)
        prediction_pairs = list(zip(batch, predictions))
        prediction_str = black.format_str(f"{prediction_pairs}", mode=black.Mode())
        LOGGER.info(f"Predicted genres:\n{prediction_str}")

        all_predictions.extend(prediction_pairs)
    return all_predictions


@click.command()
@click.option("--url", "-u", type=str, default="audioprocessor-server:8001")
@click.option("--model_name", type=str, default="ensemble_1")
@click.option("--model_input_name", type=str, default="INPUT0")
@click.option("--model_output_name", type=str, default="OUTPUT0")
@click.option("--audio_files_names", type=str, default="", required=True)
@click.option("--verbose", "-v", is_flag=True)
def run_wrapper(
    url: str,
    model_name: str,
    model_input_name: str,
    model_output_name: str,
    audio_files_names: str,
    verbose: bool = False,
):
    asyncio.run(
        main(
            url,
            model_name,
            model_input_name,
            model_output_name,
            audio_files_names,
            verbose,
        )
    )


if __name__ == "__main__":
    run_wrapper()
