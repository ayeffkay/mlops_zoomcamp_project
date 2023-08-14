import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Text

import c_python_backend_utils as c_pb_utils
import numpy as np
import preprocessing.utils as utils
import triton_python_backend_utils as pb_utils

from monitoring.monitor import DataMonitor


class TritonPythonModel:
    def initialize(self, args: Dict[Text, Text]) -> None:
        self.model_config = model_config = json.loads(args["model_config"])
        self.params = self.model_config["parameters"]
        self.input_names = self.params["input_names"]["string_value"].split(" ")
        self.output_name = self.params["output_name"]["string_value"]
        self.reserve_model_name = self.params["reserve_model_name"]["string_value"]

        input_configs = [
            pb_utils.get_input_config_by_name(model_config, input_name)
            for input_name in self.input_names
        ]
        self.input_dtypes = [
            pb_utils.triton_string_to_numpy(cfg["data_type"]) for cfg in input_configs
        ]
        output0_config = pb_utils.get_output_config_by_name(
            model_config, self.output_name
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        CUR_PATH = Path(__file__).parent.resolve()
        model_file = list(glob.glob(f"{CUR_PATH/'model.*'}"))[0]
        if os.path.splitext(model_file)[-1] == ".xgb":
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(fname=model_file)
            self.model = model
        else:
            self.model = utils.load_obj(model_file)
        self.logger = utils.init_logger(
            args["model_instance_name"], os.getenv("LOG_LEVEL", "INFO")
        )
        self.vfunc = utils.decode_str_vfunc()
        self.monitor = DataMonitor()

    async def make_request_to_another_model(self, inputs: List[Any]) -> np.ndarray:
        request = pb_utils.InferenceRequest(
            model_name=self.reserve_model_name,
            requested_output_names=[self.output_name],
            inputs=inputs,
        )
        response = await request.async_exec()
        response_np = pb_utils.get_output_tensor_by_name(
            response, self.output_name
        ).as_numpy()
        return response_np

    async def execute(
        self, requests: List[c_pb_utils.InferenceRequest]
    ) -> List[c_pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            inputs = [
                pb_utils.get_input_tensor_by_name(request, input_name)
                for input_name in self.input_names
            ]
            inputs_np = inputs[0].as_numpy()
            out_0 = self.model.predict(inputs_np)

            filenames = self.vfunc(inputs[1].as_numpy())
            self.logger.info(
                f"{len(out_0)} predictions were done! Monitoring is started..."
            )
            batch_timestamp = os.path.splitext(os.path.basename(filenames[0]))[0].split(
                "."
            )[-1]

            data_drift_flag = self.monitor(
                inputs_np, out_0, filenames, batch_timestamp, self.logger
            )
            if data_drift_flag:
                self.logger.info("Trying to make predictions with another model...")
                out_0 = await self.make_request_to_another_model([inputs[0]])
                self.logger.info("Finished prediction with another model!")

            out_tensor_0 = pb_utils.Tensor(
                self.output_name, out_0.astype(self.output0_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses
