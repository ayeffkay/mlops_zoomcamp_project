import datetime
import pickle
from numbers import Number
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import ruamel.yaml
from prefect import task
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")


def str2int_or_float(numeric_str: str) -> Number:
    if numeric_str.isdigit():
        return int(numeric_str)
    return float(numeric_str)


def str2num_from_params(params_dict: Dict[str, str]) -> Dict[str, Union[str, Number]]:
    return {
        name: (str2int_or_float(value) if value.isnumeric() else value)
        for name, value in params_dict.items()
    }


def load_obj(path: str) -> Iterable:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_obj(obj: Iterable, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_config_from_yaml(file: str) -> Dict[str, Any]:
    yaml = ruamel.yaml.YAML()
    with open(file) as f:
        params = yaml.load(f)
    return params


@task(name="encode_targets", tags=["preprocessing"], persist_result=True)
def label_encoder_from_labels(
    raw_labels: np.ndarray,
) -> LabelEncoder:
    encoder = LabelEncoder().fit(raw_labels)
    return encoder


@task(name="generate_train_and_valid_ids", tags=["preprocessing"])
def get_train_valid_ids(
    X: np.ndarray, y: np.ndarray, val_prop: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_prop, random_state=seed)
    train_val_ids = list(sss.split(X, y))
    train_ids = train_val_ids[0][0]
    val_ids = train_val_ids[0][1]
    return train_ids, val_ids
