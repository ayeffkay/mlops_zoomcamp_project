# mlops_zoomcamp_project
==============================

Final project for [MLOps zoomcamp course](https://github.com/DataTalksClub/mlops-zoomcamp). 

Disclaimer: ML is very simple here cause the goal is to try MLOps tools, not to beat SOTA :) For now the project is not completed, certain parts may not work. I hope to fix it during the second attempt.

<!-- TOC -->

- [mlops\_zoomcamp\_project](#mlops_zoomcamp_project)
  - [Task description](#task-description)
  - [Dataset](#dataset)
  - [Approach](#approach)
  - [Implemented features](#implemented-features)
    - [Experiment tracking and model registry](#experiment-tracking-and-model-registry)
    - [Workflow orchestration](#workflow-orchestration)
    - [Model deployment](#model-deployment)
    - [Model monitoring](#model-monitoring)
    - [Reproducibility](#reproducibility)
    - [Best practices](#best-practices)
    - [How to use code from repo?](#how-to-use-code-from-repo)
      - [Preparing environment](#preparing-environment)
      - [Running feature extraction](#running-feature-extraction)
      - [Running training/evaluation](#running-trainingevaluation)

<!-- /TOC -->

## Task description

The task is to define music genre by wav record. I.e., at the input we get some record and musical genre (class) is predicted at the output. For this project the following genres were determined: `{blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock}`.

## Dataset

For training and validation I took [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), this is sort of MNIST for sound. For testing some audio recordings were collected manually by myself. Due to possible usage license issues and total dataset size I will share here only the features collected (see [data](src/data) folder).

## Approach

To make task more easy the first 30 seconds from each audio were taken and widely used audio features were extracted (mel-frequency cepstral coefficients, spectral bandwidth, root mean square energy, etc.). For each vectorized feature mean and variance were computed and then tree-based classifiers trained. As dataset is balanced (100 records per genre) accuracy metric is used. 

## Implemented features

### Experiment tracking and model registry

Implemented locally via mlflow. See [training script](src/training/trainer.py) and [model registering scrip](src/training/register.py). 

### Workflow orchestration

Implemented via prefect cloud. See [feature extraction flow](src/workflows/preprocessing_workflows.py) and [orchestration script](src/workflows/orchestration.py). The last one is a draft to make deployments for different flows from python without command line.

### Model deployment

Model will be deployed with Triton Inference Server or Seldon Core, I did not have time for this :(

### Model monitoring

Will be implemented via evidently + prometheus + grafana.

### Reproducibility

Instructions are described below, suggestions are welcome.

### Best practices

pylint, mypy and black were used and [built into pre-commit hooks](.pre-commit-config.yaml)

### How to use code from repo?

#### Preparing environment

Required dependencies can be installed via pipenv

```bash
    pipenv install && pipenv shell
```

or via conda

```bash
    conda env create -n environment.yaml && conda activate mlops_zoomcamp_project
```

Before running code export need to be done

```bash
    export PYTHONPATH=$PYTHONPATH:./src
```

#### Running feature extraction

Data folder should have the following structure:

```bash
    |some_name
    --- | genre_name
        --- | genre_name.xxxx.wav
        ----| genre_name.xxxx.wav
    --- | another_genre_name
        --- | another_genre_name.xxxx.wav
        --- | another_genre_name.xxxx.wav
```

Let's pretend that `genres_original` forlder contains train and validation records, and `genres_original_eval` folder contains test records. Then feature extraction can be done as follows:

```bash
    python src/workflows/preprocessing_workflows.py --folder data/raw/genres_original --mode train --val_split_prop 0.1
    python src/workflows/preprocessing_workflows.py --folder data/raw/genres_original_eval --mode test
```

#### Running training/evaluation

Before training mlflow need to be run:

```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --serve-artifacts -p 5001
```

Then trainer can be launched

```bash
    python src/training/trainer.py
```

This script will promote model with best accuracy on the test set to the registry.
