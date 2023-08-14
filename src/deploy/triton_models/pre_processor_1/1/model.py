import json
import os
from pathlib import Path
from typing import Dict, List, Text, Tuple

import c_python_backend_utils as c_pb_utils
import numpy as np
import preprocessing.utils as utils
import triton_python_backend_utils as pb_utils
from preprocessing.feature_extractor import run_feature_extraction_in_parallel

RAW_BUCKET = os.getenv("S3_CLIENT_RAW_BUCKET", "raw-client")
FEATURES_BUCKET = os.getenv("S3_CLIENT_FEATURES_BUCKET", "features-client")
RAW_DATA = os.getenv("RAW_DATA_FOLDER", "/data/raw")
FEATURES = os.getenv("FEATURES_FOLDER", "/data/processed")


class TritonPythonModel:
    def initialize(self, args: Dict[Text, Text]) -> None:
        self.model_config = model_config = json.loads(args["model_config"])
        self.params = self.model_config["parameters"]
        self.input_name = self.params["input_name"]["string_value"]
        self.output_names = self.params["output_names"]["string_value"].split(" ")

        input0_config = pb_utils.get_input_config_by_name(model_config, self.input_name)
        self.input0_dtype = pb_utils.triton_string_to_numpy(input0_config["data_type"])

        output_configs = [
            pb_utils.get_output_config_by_name(model_config, output_name)
            for output_name in self.output_names
        ]
        self.output_dtypes = [
            pb_utils.triton_string_to_numpy(cfg["data_type"]) for cfg in output_configs
        ]
        CUR_PATH = Path(__file__).parent.resolve()
        self.audio_config_file = CUR_PATH.joinpath(
            self.params["audio_config_file"]["string_value"]
        )
        self.vfunc = utils.decode_str_vfunc()
        self.logger = utils.init_logger(
            args["model_instance_name"], os.getenv("LOG_LEVEL", "INFO")
        )

    def remove_tmp_files(self, files: List[str]):
        for file in files:
            os.remove(file)

    async def save_features_separately(
        self, files_names: List[str], features_matrix: np.ndarray
    ) -> Tuple[List[str], str]:
        feature_files = []

        batch_timestamp = str(utils.get_current_timestamp())
        for i, file_name in enumerate(files_names):
            file_name_unique = (
                f"{os.path.splitext(os.path.basename(file_name))[0]}.{batch_timestamp}"
            )
            feature_file_name = f"{FEATURES}/{file_name_unique}.pkl"
            utils.save_obj(features_matrix[i].reshape(1, -1), feature_file_name)
            feature_files.append(feature_file_name)
        self.logger.debug(f"Generated files with names {feature_files}")
        utils.put_files_by_names_to_s3(FEATURES_BUCKET, feature_files)
        self.logger.info(f"Extracted features were saved into s3:://{FEATURES_BUCKET}")
        return feature_files, batch_timestamp

    async def execute(
        self, requests: List[c_pb_utils.InferenceRequest]
    ) -> List[c_pb_utils.InferenceResponse]:
        responses = []
        if not os.path.exists(RAW_DATA):
            os.makedirs(RAW_DATA)

        if not os.path.exists(FEATURES):
            os.makedirs(FEATURES)

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(
                request, self.input_name
            ).as_numpy()

            files_names = self.vfunc(in_0)
            self.logger.info(f"Started to process {files_names}...")

            files_names = utils.get_files_by_names_from_s3(
                RAW_BUCKET, files_names, RAW_DATA
            )
            self.logger.info(f"Files was loaded from s3://{RAW_BUCKET} to {RAW_DATA}")
            out_0 = run_feature_extraction_in_parallel(
                files_names, self.audio_config_file, logger=self.logger
            )

            feature_files, _ = await self.save_features_separately(files_names, out_0)
            self.remove_tmp_files(feature_files)
            self.remove_tmp_files(files_names)

            out_1 = np.array(feature_files).reshape(-1, 1)
            self.logger.debug(f"Feature matrix shape is: {out_0.shape}")
            self.logger.debug(f"Output files shape is: {out_1.shape}")

            out_tensor_0 = pb_utils.Tensor(
                self.output_names[0], out_0.astype(self.output_dtypes[0])
            )
            out_tensor_1 = pb_utils.Tensor(
                self.output_names[1], out_1.astype(self.output_dtypes[1])
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

        return responses
