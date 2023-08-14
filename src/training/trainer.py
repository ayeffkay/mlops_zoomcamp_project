import importlib
import os
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import imblearn
import mlflow
import numpy as np
import optuna
import preprocessing.utils as utils
import sklearn
import xgboost as xgb
from mlflow.entities import ViewType
from mlflow.models import make_metric
from mlflow.tracking import MlflowClient
from optuna.integration.mlflow import MLflowCallback
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class Trainer:
    def __init__(
        self,
        tracking_uri: str,
        clf_training_func_name: Literal[
            "run_xgboost_training", "run_random_forest_training"
        ],
        train_data: List[np.ndarray],
        target_encoder: sklearn.preprocessing._label.LabelEncoder,
        features_names: List[str],
        metric_name: str = "sklearn.metrics:accuracy_score",
        val_data: Optional[List[np.ndarray]] = None,
        test_data: Optional[List[np.ndarray]] = None,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
        target_column_name: str = "genre",
        random_state: int = 42,
        n_trials_for_hyperparams: int = 10,
    ):
        self.target_encoder = target_encoder
        self.features_names = features_names
        self.train_data = train_data
        if val_data is None:
            if test_data is not None:
                val_data = deepcopy(test_data)
            else:
                test_data = deepcopy(train_data)
                val_data = deepcopy(test_data)

        train_data[1] = target_encoder.transform(train_data[1])
        val_data[1] = target_encoder.transform(val_data[1])
        test_data[1] = target_encoder.transform(test_data[1])

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.target_column_name = target_column_name
        self.classifier_kwargs = classifier_kwargs

        self.clf_name = clf_training_func_name
        self.experiment_name = clf_training_func_name
        self.clf_training_func = partial(
            getattr(Trainer, clf_training_func_name),
            feature_names_in=self.features_names,
        )

        self.metric_name, self.metric_func = Trainer.get_metric_func(metric_name)
        self.random_state = random_state
        self.n_trials = n_trials_for_hyperparams

        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.experiment = mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def geometric_mean_score(eval_df, _builtin_metrics):
        return imblearn.metrics.geometric_mean_score(
            eval_df["target"], eval_df["prediction"]
        )

    def log_numpy(self, np_data: Union[List[np.ndarray], np.ndarray], context: str):
        if isinstance(np_data, list):
            np_data = utils.concat_features_with_target(np_data)

        mlflow_dataset = mlflow.data.from_numpy(np_data)
        mlflow.log_input(mlflow_dataset, context=context)

        data_file = f"/tmp/{context}.pkl"
        utils.save_obj(np_data, data_file)
        mlflow.log_artifact(data_file, "data")
        os.remove(data_file)

    def log_inputs(self):
        self.log_numpy(np.array(self.features_names), context="features_names")
        self.log_numpy(self.train_data, context="train")
        self.log_numpy(self.val_data, context="valid")
        if self.test_data is not None:
            self.log_numpy(self.test_data, context="test")

        encoder_file = "/tmp/target_encoder.pkl"
        utils.save_obj(self.target_encoder, encoder_file)
        mlflow.log_artifact(encoder_file, "model")
        os.remove(encoder_file)

    @staticmethod
    def get_metric_func(metric_name: str) -> Tuple[str, Callable]:
        module, metric_name = metric_name.split(":")
        metric_func = getattr(importlib.import_module(module), metric_name)
        return metric_name, metric_func

    def run_eval_step(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.metric_func(y_true, y_pred)

    @staticmethod
    def make_predictions(
        model: Any,
        data: np.ndarray,
        pred_file: str,
        pred_proba_file: str,
        artifact_folder: str,
    ):
        predictions = model.predict(data)
        utils.save_obj(predictions, pred_file)
        mlflow.log_artifact(pred_file, artifact_folder)
        os.remove(pred_file)

        predictions_probas = model.predict_proba(data)
        utils.save_obj(predictions_probas, pred_proba_file)
        mlflow.log_artifact(pred_proba_file, artifact_folder)
        os.remove(pred_proba_file)

    @staticmethod
    def run_xgboost_training(
        classifier_kwargs,
        X_train,
        y_train,
        log: bool = False,
        artifact_name: str = "model",
        feature_names_in: Optional[List[str]] = None,
    ) -> xgb.sklearn.XGBClassifier:
        xgb_model = xgb.XGBClassifier(**classifier_kwargs)
        xgb_model.fit(X_train, y_train)
        if log:
            signature = mlflow.models.infer_signature(
                X_train, xgb_model.predict(X_train)
            )
            mlflow.xgboost.log_model(xgb_model, artifact_name, signature=signature)
            Trainer.make_predictions(
                xgb_model, X_train, "y_train_pred.pkl", "y_train_pred_proba.pkl", "data"
            )
        return xgb_model

    @staticmethod
    def check_kwargs(clf: Any, clf_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: value
            for name, value in clf_kwargs.items()
            if name in clf.__init__.__code__.co_varnames
        }

    @staticmethod
    def run_kneighbors_classifier_training(
        classifier_kwargs,
        X_train,
        y_train,
        log: bool = False,
        artifact_name: str = "model",
        feature_names_in: Optional[List[str]] = None,
    ):
        clf = KNeighborsClassifier(**classifier_kwargs)
        if feature_names_in:
            clf.feature_names_in_ = feature_names_in
        clf.fit(X_train, y_train)
        if log:
            signature = mlflow.models.infer_signature(X_train, clf.predict(X_train))
            mlflow.sklearn.log_model(clf, artifact_name, signature=signature)
            Trainer.make_predictions(
                clf, X_train, "y_train_pred.pkl", "y_train_pred_proba.pkl", "data"
            )
        return clf

    @staticmethod
    def run_random_forest_training(
        classifier_kwargs,
        X_train,
        y_train,
        log: bool = False,
        artifact_name: str = "model",
        feature_names_in: Optional[List[str]] = None,
    ) -> sklearn.ensemble._forest.RandomForestClassifier:
        classifier_kwargs = Trainer.check_kwargs(
            RandomForestClassifier, classifier_kwargs
        )
        rf = RandomForestClassifier(**classifier_kwargs)
        if feature_names_in:
            rf.feature_names_in_ = np.array(feature_names_in)
        rf.fit(X_train, y_train)
        if log:
            signature = mlflow.models.infer_signature(X_train, rf.predict(X_train))
            mlflow.sklearn.log_model(rf, artifact_name, signature=signature)
            Trainer.make_predictions(
                rf, X_train, "y_train_pred.pkl", "y_train_pred_proba.pkl", "data"
            )
        return rf

    def xgboost_objective(self, trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 200, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "objective": "multi:softprob",
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "num_class": len(self.target_encoder.classes_),
            "random_state": self.random_state,
        }
        clf = self.clf_training_func(params, self.train_data[0], self.train_data[1])
        y_pred = clf.predict(self.val_data[0])

        val_metric_value = self.run_eval_step(self.val_data[1], y_pred)
        return val_metric_value

    def random_forest_objective(self, trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, 10),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, 1),
            "random_state": self.random_state,
        }
        clf = self.clf_training_func(params, self.train_data[0], self.train_data[1])
        y_pred = clf.predict(self.val_data[0])

        val_metric_value = self.run_eval_step(self.val_data[1], y_pred)
        return val_metric_value

    def kneighbors_neighbors_classifier_objective(
        self, trial: optuna.trial.Trial
    ) -> float:
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 50, 5),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "leaf_size": trial.suggest_int("leaf_size", 20, 50, 5),
            "metric": trial.suggest_categorical(
                "metric",
                [
                    "minkowski",
                    "mahalanobis",
                    "canberra",
                    "cityblock",
                    "cosine",
                    "correlation",
                ],
            ),
        }
        if params["metric"] == "mahalanobis":
            params["metric_params"] = {"VI": np.cov(self.train_data[0], rowvar=False)}
        clf = self.clf_training_func(params, self.train_data[0], self.train_data[1])
        y_pred = clf.predict(self.val_data[0])

        val_metric_value = self.run_eval_step(self.val_data[1], y_pred)
        return val_metric_value

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
        elif "kneighbors" in self.clf_name:
            study.optimize(
                self.kneighbors_neighbors_classifier_objective,
                n_trials=self.n_trials,
                callbacks=[mlflc],
            )
        else:
            raise NameError("Unkwnown classifier type detected!")

        return self.get_best_hyperparams_set()

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

    def full_evaluation(
        self, data: List[np.ndarray], metric_prefix: str, model_uri: Any
    ):
        df = utils.df_from_numpy(data, self.features_names, self.target_column_name)
        res = mlflow.evaluate(
            model_uri,
            df,
            targets=self.target_column_name,
            model_type="classifier",
            evaluators=["default"],
            custom_metrics=[
                make_metric(
                    eval_fn=Trainer.geometric_mean_score,
                    greater_is_better=True,
                ),
            ],
        )
        mlflow.log_metric(
            f"{metric_prefix}_{self.metric_name}", res.metrics[self.metric_name]
        )
        return res

    def run(self):
        classifier_kwargs = (
            self.tune_hyperparams()
            if self.classifier_kwargs is None
            else self.classifier_kwargs
        )

        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=f"best_{self.experiment.name}_{utils.get_current_time()}",
        ):
            self.log_inputs()
            mlflow.log_param("target_column_name", self.target_column_name)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_params(classifier_kwargs)

            self.clf_training_func(
                classifier_kwargs,
                self.train_data[0],
                self.train_data[1],
                log=True,
                artifact_name="model",
            )
            model_uri = mlflow.get_artifact_uri("model")
            self.full_evaluation(self.val_data, "val", model_uri)
            self.full_evaluation(self.test_data, "test", model_uri)
