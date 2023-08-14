# mlops_zoomcamp_project

Final project for [MLOps zoomcamp course](https://github.com/DataTalksClub/mlops-zoomcamp) hosted by DataTalksClub.

**Disclaimer**: ML is very simple here cause the goal is to try MLOps tools, not to beat SOTA :) For now the project is not completed, certain parts may not work. Sorry for this, I hope to fix it during the second attempt.

**WARNING** After code updates README is not actual now.
<!-- TOC -->

- [mlops\_zoomcamp\_project](#mlops_zoomcamp_project)
  - [Task description](#task-description)
  - [Dataset](#dataset)
  - [Approach](#approach)
  - [Metric](#metric)
  - [Instructions](#instructions)
    - [Preparing environment](#preparing-environment)
    - [Feature extraction](#feature-extraction)
    - [Training](#training)
  - [Implemented features](#implemented-features)
    - [Experiment tracking and model registry](#experiment-tracking-and-model-registry)
    - [Workflow orchestration](#workflow-orchestration)
    - [Model deployment](#model-deployment)
    - [Model monitoring](#model-monitoring)
    - [Reproducibility](#reproducibility)
    - [Best practices](#best-practices)

<!-- /TOC -->


![Coverage](.github/badges/jacoco.svg)

## Task description

The task is to define music genre by wav record. I.e., at the input we get some record and musical genre (class) is predicted at the output. For this project the following genres were determined: `{blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock}`.

## Dataset

For training and validation I took [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), this is sort of MNIST for sound. Then some genres were supplemented by data from [ISMIR dataset](https://www.upf.edu/web/mtg/ismir2004-genre).

For testing some audio recordings were collected manually by myself (to discover data drift). I provided only part of the data in the repo because audio files take up a lot of disk space (but you can download full version from [here](https://drive.google.com/file/d/1sPptNqohrdEEvsABLGuTRmnFakVuk3SW/view?usp=sharing) or use your own audios - code is designed to work with `wav`, `mp3` and `mp4`). Note that data folder should have the following structure (or `.mp3`, `.mp4` instead of `.wav`):

```bash
    |some_name
    --- | genre_name
        --- | genre_name.xxxx.wav
        ----| genre_name.xxxx.wav
    --- | another_genre_name
        --- | another_genre_name.xxxx.wav
        --- | another_genre_name.xxxx.wav
```

Part of the dataset is located in the [data/raw](data/raw) folder, extracted features from the whole dataset are located in the [data/processed](data/processed/) folder.

## Approach

This task can be considered as multiclass classification. To make things easier, 30 seconds were randomly selected from each audio and widely used audio features were extracted (mel-frequency cepstral coefficients, spectral bandwidth, root mean square energy, etc.). For each vectorized feature mean and variance were calculated and three types of classifiers trained (RandomForest, XGBoost and KNearestNeighbors).

## Metric

GTZAN dataset was balanced (100 records per genre), but after some classes have been supplemented with data from ISMIR, dataset was no longer balanced. So to measure model performance [G-Mean Score](http://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.geometric_mean_score.html) was used.

## Instructions

### Preparing environment

Prefect cloud is used to implement orchestation, so first you should register/login into [Prefect Cloud](https://app.prefect.cloud), [create workspace](https://app.prefect.cloud/workspaces/create), work pool with name `mlops-zoomcamp` inside workspace and finally [access key](https://app.prefect.cloud/my/api-keys) Then generated workspace and access key should be assigned to `ENV PREFECT_KEY` and `ENV PREFECT_WORKSPACE` in [Dockerfile.audio](dockerfiles/Dockerfile.audio) and [Dockerfile.tritonclient](dockerfiles/Dockerfile.tritonclient)

After that all required services can be started via docker-compose:

```bash
    docker compose -f docker-compose.yaml up --build
```

or via make

```bash
    make build_and_up
```

To run feature extraction and training, go inside `audioprocessor_dev` container:

```bash
    docker exec -it audioprocessor_dev bash
```

### Feature extraction

To extract features from audios run:

```bash
    python preprocessing/preprocessing_workflows.py --input_folder /data/raw/train_valid --output_folder /data/processed --mode train --val_split_prop 0.1
    python preprocessing/preprocessing_workflows.py --input_folder /data/raw/test --output_folder /data/processed --mode test
```

If you use truncated dataset version, delete `--valid_split_prop` (provided data subset is not enough to make stratified split, this will cause errors). **Test dataset should be used both for validation and test** (in the case of truncated dataset version).

Data will be stored locally into `/data/processed` folder and into s3 bucket (`s3://audioprocessor/`). Note that feature extraction is parallelized with `prefect.dask` to speed-up pre-processing and this may be resource-hungry.

### Training

To run training use:

```bash
    python training/trainer.py
```

**Don't forget to replace val.pkl by test.pkl if valid file was not generated**. This script will tune hyperparameters for random forest and xgboost classifier, then model with best test accuracy will be promoted to the registry.


## Implemented features

### Experiment tracking and model registry

Implemented locally via mlflow. See [training script](src/training/trainer.py) and [model registering scrip](src/training/register.py).

### Workflow orchestration

Implemented via prefect cloud. See [feature extraction flow](src/preprocessing/preprocessing_workflows.py) and [orchestration script](src/workflows/orchestration.py). The last one is a draft to make deployments for different flows from python without command line.

### Model deployment

Model will be deployed with Triton Inference Server or Seldon Core, I did not have time for this :(

### Model monitoring

Will be implemented via evidently + prometheus + grafana.

### Reproducibility

Instructions are described, suggestions are welcome.

### Best practices

pylint, mypy and black were used and [built into pre-commit hooks](.pre-commit-config.yaml). To check pre-commit hooks, use pre-commit:

```bash
    pip install pre-commit
    pre-commit install
```
