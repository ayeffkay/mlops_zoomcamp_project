import datetime
import logging
import os
from typing import List

import numpy as np
import psycopg
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite

import preprocessing.utils as utils

DATA_MONITORING_DB = os.getenv("DATA_MONITORING_DB", "data_monitoring")
DATA_MONITORING_TABLE = os.getenv("DATA_MONITORING_TABLE", "test")
FEATURES_BUCKET_NAME = os.getenv("S3_FEATURES_BUCKET", "features")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_HOST_NAME = os.getenv("DB_HOST", "db")
PG_PORT = int(os.getenv("DB_PORT", "5432"))


class DataMonitor:
    target_name = "genre"
    data_files = utils.get_files_by_names_from_s3(
        "features",
        ["/data/processed/feature_names.pkl", "/data/processed/train.pkl"],
        save_to="./data_artifacts",
    )
    feature_names = utils.load_obj(data_files[0])
    train_data = utils.load_obj(data_files[1])
    reference_data = utils.df_from_numpy(train_data, feature_names, target_name)
    column_mapping = ColumnMapping(
        prediction=target_name, numerical_features=feature_names, target=target_name
    )
    report = Report(
        metrics=[
            DatasetDriftMetric(columns=feature_names),
        ]
    )
    create_table_statement = f"""
    drop table if exists {DATA_MONITORING_TABLE};
    create table {DATA_MONITORING_TABLE}(
        timestamp timestamp,
        drift_share float,
        number_of_columns integer,
        number_of_drifted_columns integer,
        share_of_drifted_columns float,
        dataset_drift boolean
    )
    """
    drift_report = Report(metrics=[DataDriftPreset()])
    test_suite = TestSuite(tests=[DataDriftTestPreset()])
    connect_str = (
        f"host={PG_HOST_NAME} port={PG_PORT} user={PG_USER} password={PG_PASSWORD}"
    )
    db_connect_str = f"""host={PG_HOST_NAME} port={PG_PORT}
                    dbname={DATA_MONITORING_DB}
                    user={PG_USER} password={PG_PASSWORD}"""

    def __init__(self):
        with psycopg.connect(self.connect_str, autocommit=True) as conn:
            res = conn.execute(
                f"SELECT 1 FROM pg_database WHERE datname='{DATA_MONITORING_DB}'"
            )
            if len(res.fetchall()) == 0:
                conn.execute(f"create database {DATA_MONITORING_DB};")
            with psycopg.connect(self.db_connect_str) as conn:
                conn.execute(self.create_table_statement)

    def __call__(
        self,
        new_data: np.ndarray,
        predictions: np.ndarray,
        files_names: List[str],
        batch_timestamp: str,
        logger: logging.Logger,
    ) -> bool:
        new_df = utils.df_from_numpy(
            [new_data, predictions], self.feature_names, self.target_name
        )
        self.report.run(
            reference_data=self.reference_data,
            current_data=new_df,
            column_mapping=self.column_mapping,
        )

        result = self.report.as_dict()["metrics"][0]["result"]
        result["timestamp"] = datetime.datetime.fromtimestamp(int(batch_timestamp))
        result["DATA_MONITORING_TABLE"] = DATA_MONITORING_TABLE

        conn = psycopg.connect(self.db_connect_str, autocommit=True)
        cur = conn.cursor()
        cur.execute(
            """insert into {DATA_MONITORING_TABLE}(timestamp, drift_share,
            number_of_columns, number_of_drifted_columns,
            share_of_drifted_columns, dataset_drift) values
            ('{timestamp}', {drift_share}, {number_of_columns},
            {number_of_drifted_columns}, {share_of_drifted_columns},
            {dataset_drift})""".format(
                **result
            )
        )
        conn.commit()
        cur.close()
        conn.close()

        dataset_drift_flag = result["dataset_drift"]
        if dataset_drift_flag:
            logger.info(f"Data drift detected for files {files_names}!")
            # debugging dashboards
            # self.test_suite.run(reference_data=self.reference_data, current_data=new_df, column_mapping=self.column_mapping)
            # self.drift_report.run(reference_data=self.reference_data, current_data=new_df, column_mapping=self.column_mapping)
        return dataset_drift_flag
