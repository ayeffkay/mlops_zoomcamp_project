# mlops_zoomcamp_project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Codecov](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ayeffkay/c1336d0706263813a2eabbb344302bc1/raw/codecov.json)

Final project for [MLOps zoomcamp course](https://github.com/DataTalksClub/mlops-zoomcamp) hosted by DataTalksClub.

**Disclaimer**: ML is very simple here cause the goal is to try MLOps tools, not to beat SOTA :)

<!-- TOC -->

- [mlops\_zoomcamp\_project](#mlops_zoomcamp_project)
  - [Task description](#task-description)
    - [Dataset](#dataset)
    - [Approach](#approach)
    - [Metric](#metric)
  - [System architecture](#system-architecture)
  - [Instructions](#instructions)
    - [Prerequisites](#prerequisites)
    - [Code quality checks](#code-quality-checks)
    - [Launching services](#launching-services)
    - [Running feature extraction, training and registering model](#running-feature-extraction-training-and-registering-model)
  - [Implemented features](#implemented-features)
    - [Experiment tracking and model registry](#experiment-tracking-and-model-registry)
    - [Workflow orchestration](#workflow-orchestration)
    - [Model deployment](#model-deployment)
    - [Model monitoring](#model-monitoring)
    - [Reproducibility](#reproducibility)
    - [Best practices](#best-practices)

<!-- /TOC -->

## Task description

The task is to define music genre by wav record. I.e., at the input we get some record and musical genre (class) is predicted at the output. For this project the following genres were determined: `{blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock}`.

### Dataset

For training and validation I took [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), this is sort of MNIST for sound. Then some genres were supplemented by records from [ISMIR dataset](https://www.upf.edu/web/mtg/ismir2004-genre).

For testing some audio recordings were collected manually by myself (to discover data drift). I provided only [part of the training data](data/raw/genres_original_subset), [cropped test audios](data/raw/genres_original_eval) and [cropped random audios without labels](data/raw/random_data_cut/) in the repo because audio files take up a lot of disk space, also working with full dataset version is resource-hungry and requires for about an hour of free time (depending on hardware). But if you wish, full dataset for training can be downloaded from [here](https://drive.google.com/file/d/1sPptNqohrdEEvsABLGuTRmnFakVuk3SW/view?usp=sharing). Also you can use your own audios - code is designed to work with `wav`, `mp3` and `mp4`. Note that data folder should have the following structure (or `.mp3`, `.mp4` instead of `.wav`), e.g.:

```bash
    | data_folder_name (e.g., genres_original_subset)
    --- | genre_name
        --- | genre_name.xxxxx.wav
        ----| genre_name.xxxxx.wav
    --- | another_genre_name
        --- | another_genre_name.xxxxx.wav
        --- | another_genre_name.xxxxx.wav
```

Extracted features from the whole dataset are located in the [data/processed](data/processed/) folder. These features & [target encoder](data/processed/target_encoder.pkl) are enough to train ML models.

### Approach

The task of genre prediction can be considered as multiclass classification. To make things easier, 30 seconds were randomly selected from each audio and widely used audio features were extracted (mel-frequency cepstral coefficients, spectral bandwidth, root mean square energy, etc.). For each vectorized feature mean and variance were calculated and three types of classifiers trained (RandomForest, XGBoost and KNearestNeighbors).

### Metric

GTZAN dataset was balanced (100 records per genre), but after some classes have been supplemented with data from ISMIR, dataset was no longer balanced. So to measure model performance [G-Mean Score](http://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.geometric_mean_score.html) was used.

## System architecture

TBD

## Instructions

### Prerequisites

The system is deployed with docker and can be run locally with docker-compose. If you don't have docker-compose, check [these instructions](https://docs.docker.com/compose/install/). Also you will need to get prefect account before running docker-compose because prefect cloud is used to implement orchestration. First you should register/login into [Prefect Cloud](https://app.prefect.cloud), [create workspace](https://app.prefect.cloud/workspaces/create), work pool with name `mlops-zoomcamp` inside workspace and finally create [access key](https://app.prefect.cloud/my/api-keys). Then generated workspace and access key should be assigned to `ENV PREFECT_KEY` and `ENV PREFECT_WORKSPACE` in [Dockerfile.audio](dockerfiles/Dockerfile.audio),[Dockerfile.tritonclient](dockerfiles/Dockerfile.tritonclient) and [prefect.env](docker_env/prefect.env).

### Code quality checks

```bash
    make setup
    make quality_checks
```

### Launching services

All required services can be started with [docker-compose file](docker-compose.yaml):

```bash
    make build_and_up
```

Note that there are 9 services inside docker-compose, so they will be pulled and built for about an hour.

### Running feature extraction, training and registering model

TBD

## Implemented features

### Experiment tracking and model registry

Implemented locally via mlflow.

### Workflow orchestration

Implemented via prefect cloud.

### Model deployment

Two best of registered models were deployed in semi-automatic mode with Triton Inference Server with python backend. I tried onnxruntime backend, but it fails for sklearn models.

### Model monitoring

Implemented with evidently & postgresql (data drift) + prometheus (service metrics from Triton) + grafana (displays nice plots and sending alerts for data drift cases). If data drift occurs predictions are generated with second model.

### Reproducibility

Instructions are described, suggestions are welcome.

### Best practices

- pylint, mypy and black were used and [built into pre-commit hooks](.pre-commit-config.yaml)
- makefiles
- unit tests
- CI/CD
