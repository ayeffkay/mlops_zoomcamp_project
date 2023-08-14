import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Text

import c_python_backend_utils as c_pb_utils
import preprocessing.utils as utils
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args: Dict[Text, Text]) -> None:
        self.model_config = model_config = json.loads(args["model_config"])
        self.params = self.model_config["parameters"]
        self.input_name = self.params["input_name"]["string_value"]
        self.output_name = self.params["output_name"]["string_value"]

        input0_config = pb_utils.get_input_config_by_name(model_config, self.input_name)
        self.input0_dtype = pb_utils.triton_string_to_numpy(input0_config["data_type"])
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

    async def execute(
        self, requests: List[c_pb_utils.InferenceRequest]
    ) -> List[c_pb_utils.InferenceResponse]:
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(
                request, self.input_name
            ).as_numpy()
            out_0 = self.model.predict(in_0)
            self.logger.info(f"{len(out_0)} predictions were done!")

            out_tensor_0 = pb_utils.Tensor(
                self.output_name, out_0.astype(self.output0_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses
