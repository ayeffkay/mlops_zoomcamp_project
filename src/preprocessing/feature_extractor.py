import importlib
import logging
import multiprocessing as mp
from collections import ChainMap
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import sklearn
from prefect import task

import preprocessing.utils as utils
from preprocessing.audio import Audio


class Feature:
    def __init__(
        self,
        feature_func_name: str,
        kwargs_names: List[str],
        add_zero_axis: bool = False,
        min_max_scaling: bool = False,
        complex_to_real: bool = False,
    ):
        module_name, func_name = feature_func_name.split(":")
        self.feat_name = func_name
        self.feat_func = getattr(
            importlib.import_module(
                module_name,
            ),
            func_name,
        )
        self.add_zero_axis = add_zero_axis
        self.complex_to_real = complex_to_real
        self.min_max_scaling = (
            sklearn.preprocessing.minmax_scale if min_max_scaling else None
        )
        self.kwargs_names = kwargs_names

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def _kwargs_from_audio(self, audio: Audio) -> Dict[str, float]:
        kwargs_dict = {}
        for kwarg_name in self.kwargs_names:
            if kwarg_name == "y":
                kwargs_dict[kwarg_name] = audio.audio
            elif kwarg_name == "sr":
                kwargs_dict[kwarg_name] = audio.sampling_rate
            elif kwarg_name in ["n_fft", "frame_length"]:
                kwargs_dict[kwarg_name] = audio.frame_length
            else:
                kwargs_dict[kwarg_name] = getattr(audio, kwarg_name)
        return kwargs_dict

    def __call__(self, audio: Audio) -> Dict[str, float]:
        kwargs_dict = self._kwargs_from_audio(audio)
        res = self.feat_func(**kwargs_dict)
        if isinstance(res, tuple):
            res = res[0]
        if isinstance(res, np.ndarray):
            if self.complex_to_real:
                res = np.abs(res)
            if self.add_zero_axis:
                res = res.reshape(1, -1)
            if self.min_max_scaling:
                res = self.min_max_scaling(res, axis=1)
            mu = np.mean(res, axis=1)
            var = np.var(res, axis=1)
            feature_dicts = [
                {f"{self.feat_name}_mean_{i}": m, f"{self.feat_name}_var_{i}": v}
                for i, (m, v) in enumerate(zip(mu, var))
            ]
            feature_dict = dict(ChainMap(*feature_dicts))
        else:
            feature_dict = {self.feat_name: res}
        return feature_dict


class FeatureExtractor:
    _feature_extractors_config = [
        ["librosa.feature:mfcc", ["y", "sr", "n_mfcc"], False, False, True],
        ["librosa.beat:beat_track", ["y", "sr"], False, False, False],
        ["librosa.effects:harmonic", ["y"], True, False, False],
        ["librosa.effects:percussive", ["y"], True, False, False],
        [
            "librosa.feature:spectral_bandwidth",
            ["y", "sr", "n_fft", "hop_length"],
            False,
            True,
            False,
        ],
        ["librosa.feature:spectral_rolloff", ["y", "sr"], False, True, False],
        [
            "librosa.feature:rms",
            ["y", "frame_length", "hop_length"],
            False,
            False,
            False,
        ],
    ]
    _feature_extractors = [
        Feature.from_config(*cfg) for cfg in _feature_extractors_config
    ]

    def __call__(
        self,
        audio: Audio,
        label_encoder: Optional[sklearn.preprocessing._label.LabelEncoder] = None,
    ) -> Dict[str, Union[str, float]]:
        features = [
            feature_extractor(audio) for feature_extractor in self._feature_extractors
        ]
        features_dict = dict(ChainMap(*features))
        if label_encoder is not None:
            features_dict["genre"] = label_encoder.transform(np.array([audio.genre]))[0]
        else:
            features_dict["genre"] = audio.genre
        return features_dict


@task(name="feature_extraction_from_file", log_prints=True, tags=["preprocessing"])
def run_feature_extraction_from_audio_file(
    file: str,
    audio_config_file: str,
    to_pandas: bool = False,
    feature_extractor: Optional[FeatureExtractor] = None,
    queue: Optional[mp.queues.Queue] = None,
    semaphore: Optional[mp.synchronize.Semaphore] = None,
    logger: Optional[logging.Logger] = None,
) -> Union[pd.DataFrame, Sequence[Any]]:
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    audio_config = utils.load_config_from_yaml(audio_config_file)
    audio_config["audio_file"] = file
    audio = Audio(**audio_config)
    # failed to load audio, return nothing
    if audio.audio is None:
        print(f"Skipped {file} (error while loading)")
        return []
    feature_dict: Sequence[Any] = [feature_extractor(audio)]
    log_msg = f"Features extracted from file {file}"
    if logger is not None:
        logger.info(log_msg)
    else:
        print(log_msg)
    # semaphore and queue were implemented for multiprocessing
    # on server side during deploy stage
    if semaphore is not None:
        semaphore.release()
    if queue is not None:
        queue.put(feature_dict)
    if to_pandas:
        res = pd.DataFrame(data=feature_dict, columns=feature_dict[0].keys())
        return res
    return feature_dict


def run_feature_extraction_in_parallel(
    all_files: List[str],
    audio_config_file: str,
    target_column_name: str = "genre",
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    feature_extractor = FeatureExtractor()
    queue: mp.queues.Queue = mp.Queue()
    semaphore: mp.synchronize.Semaphore = mp.Semaphore(mp.cpu_count() - 2)

    procs = []
    for file in all_files:
        semaphore.acquire()
        proc = mp.Process(
            target=run_feature_extraction_from_audio_file.fn,
            args=(
                file,
                audio_config_file,
                False,
                feature_extractor,
                queue,
                semaphore,
                logger,
            ),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    feature_dicts: List[dict] = []
    while not queue.empty():
        feature_dicts.extend(queue.get())

    X = pd.DataFrame(feature_dicts, columns=feature_dicts[0].keys())
    X = X.drop(target_column_name, axis=1)
    X_np = X.values
    return X_np
