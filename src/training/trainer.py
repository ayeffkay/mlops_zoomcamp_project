import importlib
from typing import Any, Dict, List, Literal, Optional, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
import preprocessing.utils as utils
import sklearn
import xgboost as xgb
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from optuna.integration.mlflow import MLflowCallback
from optuna.samplers import TPESampler
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from training.register import register_best_model


class Trainer:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        clf_training_func_name: Literal[
            "run_xgboost_training", "run_random_forest_training"
        ],
        train_data: Union[List[np.ndarray], str],
        target_encoder: Union[sklearn.preprocessing._label.LabelEncoder, str],
        features_names: Union[List[str], str],
        metric_name: str = "accuracy_score",
        val_data: Optional[Union[List[np.ndarray], str]] = None,
        test_data: Optional[Union[List[np.ndarray], str]] = None,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
        target_column_name: str = "genre",
        random_state: int = 42,
        n_trials_for_hyperparams: int = 10,
        val_prop: float = 0.1,
    ):
        if isinstance(target_encoder, str):
            target_encoder = utils.load_obj(target_encoder)
        self.target_encoder = target_encoder

        if isinstance(features_names, str):
            features_names = utils.load_obj(features_names)
        self.features_names = features_names

        if isinstance(train_data, str):
            train_data = list(utils.load_obj(train_data))
        if val_data is None:
            train_ids, val_ids = utils.get_train_valid_ids(
                train_data[0], train_data[1], val_prop, random_state
            )
            train_data, val_data = train_data[train_ids], train_data[val_ids]
        elif isinstance(val_data, str):
            val_data = list(utils.load_obj(val_data))

        train_data[1] = target_encoder.transform(train_data[1])
        val_data[1] = target_encoder.transform(val_data[1])
        self.train_data = train_data
        self.val_data = val_data

        self.test_data = None
        if test_data is not None and isinstance(test_data, str):
            test_data = list(utils.load_obj(test_data))
            test_data[1] = target_encoder.transform(test_data[1])
            self.test_data = test_data

        self.target_column_name = target_column_name
        self.classifier_kwargs = classifier_kwargs

        self.clf_name = clf_training_func_name
        self.clf_training_func = getattr(__class__, clf_training_func_name)

        self.metric_name = metric_name
        self.random_state = random_state
        self.n_trials = n_trials_for_hyperparams

        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        self.experiment = mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def concat_features_with_target(np_data: List[np.ndarray]) -> np.ndarray:
        return np.hstack((np_data[0], np_data[1].reshape(-1, 1)))

    def log_numpy(self, np_data: Union[List[np.ndarray], np.ndarray], context: str):
        if isinstance(np_data, list):
            np_data = __class__.concat_features_with_target(np_data)

        mlflow_dataset = mlflow.data.from_numpy(np_data)
        mlflow.log_input(mlflow_dataset, context=context)

    # @task(name="log_training_inputs", tags=["training"])
    def log_inputs(self):
        self.log_numpy(self.train_data, context="train")
        self.log_numpy(self.val_data, context="valid")
        if self.test_data is not None:
            self.log_numpy(self.test_data, context="test")

        utils.save_obj(self.target_encoder, "target_encoder.pkl")
        mlflow.log_artifact("target_encoder.pkl", "target_encoder")

    @staticmethod
    def run_eval_step(
        y_true: np.ndarray, y_pred: np.ndarray, metric_name: str
    ) -> float:
        metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
        return metric(y_true, y_pred)

    @staticmethod
    # @task(name="xgboost_classifier_training", tags=["training"])
    def run_xgboost_training(
        classifier_kwargs, X_train, y_train, log: bool = False
    ) -> xgb.sklearn.XGBClassifier:
        xgb_model = xgb.XGBClassifier(**classifier_kwargs)
        xgb_model.fit(X_train, y_train)
        if log:
            signature = mlflow.models.infer_signature(
                X_train, xgb_model.predict(X_train)
            )
            mlflow.xgboost.log_model(xgb_model, "model", signature=signature)
        return xgb_model

    @staticmethod
    # @task(name="random_forest_classifier_training", tags=["training"])
    def run_random_forest_training(
        classifier_kwargs, X_train, y_train, log: bool = False
    ) -> sklearn.ensemble._forest.RandomForestClassifier:
        rf = RandomForestClassifier(**classifier_kwargs)
        rf.fit(X_train, y_train)
        if log:
            signature = mlflow.models.infer_signature(X_train, rf.predict(X_train))
            mlflow.sklearn.log_model(rf, "model", signature=signature)
        return rf

    def xgboost_objective(self, trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "objective": "multi:softmax",
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "num_class": len(self.target_encoder.classes_),
            "random_state": self.random_state,
        }
        clf = self.clf_training_func(params, self.train_data[0], self.train_data[1])
        y_pred = clf.predict(self.val_data[0])

        val_metric_value = __class__.run_eval_step(
            self.val_data[1], y_pred, self.metric_name
        )
        return val_metric_value

    def random_forest_objective(self, trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, 1),
            "random_state": self.random_state,
        }
        clf = self.clf_training_func(params, self.train_data[0], self.train_data[1])
        y_pred = clf.predict(self.val_data[0])

        val_metric_value = __class__.run_eval_step(
            self.val_data[1], y_pred, self.metric_name
        )
        return val_metric_value

    # @task(name="tune_hyperparams", tags=["training"])
    def tune_hyperparams(self):
        current_experiment = mlflow.get_experiment_by_name(self.experiment_name)
        experiment_id = current_experiment.experiment_id

        mlflc = MLflowCallback(
            tracking_uri=self.tracking_uri,
            metric_name=f"val_{self.metric_name}",
            mlflow_kwargs={
                "experiment_id": experiment_id,
                "run_name": f"{self.experiment_name}_optuna_{utils.get_current_time()}",
            },
        )

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        if "random_forest" in self.clf_name:
            study.optimize(
                self.random_forest_objective, n_trials=self.n_trials, callbacks=[mlflc]
            )
        elif "xgboost" in self.clf_name:
            study.optimize(
                self.xgboost_objective, n_trials=self.n_trials, callbacks=[mlflc]
            )
        else:
            raise NameError("Unkwnown classifier type detected!")

        return self.get_best_hyperparams_set()

    # @task(name="get_best_hyperparams_set", tags=["training"])
    def get_best_hyperparams_set(self) -> Dict[str, Any]:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=[f"metrics.val_{self.metric_name} DESC"],
        )[0]
        best_params = utils.str2num_from_params(best_run.data.params)
        return best_params

    @staticmethod
    def df_from_numpy(
        np_data: List[np.ndarray],
        features_names: List[str],
        target_name: str,
    ) -> pd.DataFrame:
        np_data = __class__.concat_features_with_target(np_data)
        columns = features_names + [target_name]
        return pd.DataFrame(np_data, columns=columns)

    # @flow(name="full_eval_after_train", validate_parameters=False)
    def full_evaluation(
        self, data: List[np.ndarray], metric_prefix: str, model_uri: Any
    ):
        df = __class__.df_from_numpy(data, self.features_names, self.target_column_name)
        res = mlflow.evaluate(
            model_uri,
            df,
            targets=self.target_column_name,
            model_type="classifier",
            evaluators=["default"],
        )
        mlflow.log_metric(
            f"{metric_prefix}_{self.metric_name}", res.metrics[self.metric_name]
        )
        return res

    def __call__(self):
        if self.classifier_kwargs is None:
            self.classifier_kwargs = self.tune_hyperparams()
        try:
            self.experiment = mlflow.set_experiment(self.experiment_name)
        except:
            self.experiment = mlflow.create_experiment(self.experiment_name)

        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=f"best_{self.experiment.name}_{utils.get_current_time()}",
        ):
            self.log_inputs()
            mlflow.log_param("target_column_name", self.target_column_name)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_params(self.classifier_kwargs)

            clf = self.clf_training_func(
                self.classifier_kwargs, self.train_data[0], self.train_data[1], log=True
            )
            model_uri = mlflow.get_artifact_uri("model")
            val_res = self.full_evaluation(self.val_data, "val", model_uri)

            test_res = (
                self.full_evaluation(self.test_data, "test", model_uri)
                if self.test_data is not None
                else None
            )
        return clf, val_res, test_res


if __name__ == "__main__":
    local_tracking_uri = "http://mlflow:5001"
    trainer_1 = Trainer(
        tracking_uri=local_tracking_uri,
        experiment_name="xgboost",
        clf_training_func_name="run_xgboost_training",
        train_data="/data/processed/train.pkl",
        val_data="/data/processed/val.pkl",
        test_data="/data/processed/test.pkl",
        target_column_name="genre",
        target_encoder="/data/processed/target_encoder.pkl",
        features_names="/data/processed/feature_names.pkl",
        metric_name="accuracy_score",
    )
    trainer_1()

    trainer_2 = Trainer(
        tracking_uri=local_tracking_uri,
        experiment_name="random_forest",
        clf_training_func_name="run_random_forest_training",
        train_data="/data/processed/train.pkl",
        val_data="/data/processed/val.pkl",
        test_data="/data/processed/test.pkl",
        target_column_name="genre",
        target_encoder="/data/processed/target_encoder.pkl",
        features_names="/data/processed/feature_names.pkl",
        metric_name="accuracy_score",
    )
    trainer_2()

    register_best_model(
        local_tracking_uri,
        ["random_forest", "xgboost"],
        "best_model",
        "test_accuracy_score",
    )
