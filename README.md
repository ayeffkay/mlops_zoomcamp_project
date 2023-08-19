# mlops_zoomcamp_project
![Project stack](images/stack.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/ayeffkay/mlops_zoomcamp_project/graph/badge.svg?token=PDIYISF8JT)](https://codecov.io/gh/ayeffkay/mlops_zoomcamp_project)

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
    - [Launching services](#launching-services)
    - [Tests](#tests)
    - [Feature extraction](#feature-extraction)
      - [First way](#first-way)
      - [Second way](#second-way)
      - [Third way](#third-way)
    - [Traning models](#traning-models)
    - [Model registry](#model-registry)
    - [Models deployment](#models-deployment)
      - [Inference description](#inference-description)
      - [Preparing models' environment](#preparing-models-environment)
      - [Running inference](#running-inference)
    - [Models monitoring](#models-monitoring)
  - [Implemented features: summary](#implemented-features-summary)

<!-- /TOC -->

## Task description

The task is to define music genre by wav record. I.e., at the input we get some record and musical genre (class) is predicted at the output. For this project the following genres were determined: `{blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock}`.

### Dataset

For training and validation I took [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), this is sort of MNIST for sound. Then some genres were supplemented by records from [ISMIR dataset](https://www.upf.edu/web/mtg/ismir2004-genre).

For testing some audio recordings were collected manually by myself (to discover data drift). I provided only [piece of the training data](data/raw/genres_original_subset), [cropped test audios](data/raw/genres_original_eval) and [cropped random audios without labels](data/raw/random_data_cut/) in the repo because audio files take up a lot of disk space, also working with full dataset version is resource-hungry and requires for about an hour of free time (depending on hardware). But if you wish, full dataset for training can be downloaded from [here](https://drive.google.com/file/d/1sPptNqohrdEEvsABLGuTRmnFakVuk3SW/view?usp=sharing). Also you can use your own audios - code is designed to work with `wav`, `mp3` and `mp4`. Note that data folder should have the following structure (or `.mp3`, `.mp4` instead of `.wav`), e.g.:

```bash
    | data_folder_name (e.g., genres_original_subset)
    --- | genre_name
        --- | genre_name.xxxxx.wav
        ----| genre_name.xxxxx.wav
    --- | another_genre_name
        --- | another_genre_name.xxxxx.wav
        --- | another_genre_name.xxxxx.wav
```

Extracted features from the whole dataset are located in the [data/processed](data/processed/) folder. These features plus [target encoder](data/processed/target_encoder.pkl) are enough to train ML models.

### Approach

The task of genre prediction can be considered as multiclass classification. To make things easier, 30 seconds were randomly selected from each audio and widely used audio features were extracted (mel-frequency cepstral coefficients, spectral bandwidth, root mean square energy, etc.). For each vectorized feature mean and variance were calculated and three types of classifiers trained ([RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRFClassifier) and [KNearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)).

### Metric

GTZAN dataset was balanced (100 records per genre), but after some classes have been supplemented with data from ISMIR, dataset was no longer balanced. So to measure model performance [G-Mean Score](http://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.geometric_mean_score.html) was used.

## System architecture

TBD

## Instructions

### Prerequisites

The system is deployed with docker and can be run locally using docker-compose. If you don't have docker-compose, check [these instructions](https://docs.docker.com/compose/install/). Also you will need to create prefect account before running docker-compose (as prefect cloud is used to implement orchestration). First you should register/login into [Prefect Cloud](https://app.prefect.cloud), [create workspace](https://app.prefect.cloud/workspaces/create), work pool with name you want inside workspace and finally get [access key](https://app.prefect.cloud/my/api-keys). Then generated access key, pool and workspace names should be assigned to `PREFECT_KEY`, `PREFECT_POOL` and `PREFECT_WORKSPACE` in [prefect.env](docker_env/prefect.env).

If you will work with full dataset version (this is not required for code testing), download it from [here](https://drive.google.com/file/d/1sPptNqohrdEEvsABLGuTRmnFakVuk3SW/view?usp=sharing) and save into `src/data/raw`. Check if downloaded folder `genres_original` has the same structure as [genres_original_subset](data/raw/genres_original_subset/).

### Launching services

All required services can be started with [docker-compose file](docker-compose.yaml):

```bash
    make build_and_up
```

Note that there are 9 services inside docker-compose, so they will be pulled and built for about an hour if you don't have any of them.
For development/checking code go inside container `audioprocessor_dev` (you will need to create 2 separate windows with this container):

```bash
  docker exec -it audioprocessor_dev bash
```

Then run the following steps inside container:

```bash
  make create_buckets run_prefect
```

`create_buckets` step will create buckets for raw data and for extracted features, `run_prefect` step will make authorization and start pool afterwards. After prefect agent start, switch to the second window of `audioprocessor_dev` container. Then all commands below must be executed inside this container.

### Tests

Code quality and health can be checked either locally (outside the container)

```bash
    make setup
    make quality_checks unit_tests
```

or from `audioprocessor_dev` container

```bash
  make setup
  make quality_checks unit_tests
```

To check code quality before commits run outside the container

```bash
  make pre_commit
```

### Feature extraction

You have three ways here:

- Run full feature extraction (resource-hungry, needs about an hour depending on the hardware). This step requires full dataset version, download it from Google Drive by the link provided above
- Run feature extraction on [piece of the dataset](data/raw/genres_original_subset/) just for testing purposes
- Use [extracted features from the full dataset](data/processed/)

Feature extraction is implemented as individual prefect deployment, and uses [prefect-dask](https://docs.prefect.io/2.11.4/guides/dask-ray-task-runners/) for parallelization. In practice we don't need feature extraction and training models together, instead more frequent use case is to extract features once and make experiments with different models for pre-defined feature set.

Progress can be tracked in the terminal with prefect agent. After the first run this deployment can be launched from prefect deployments UI.

#### First way

Check if `/data/raw/genres_original` folder exists inside container and is not empty. Then run feature extraction pipelines:

```bash
  # put wav files to s3
  make put_raw_train_data_to_s3 put_raw_test_data_to_s3
  # run feature extraction and save to features bucket
  make preprocess_raw_train_data preprocess_raw_test_data
```

#### Second way

```bash
  # put wav files to s3
  make put_raw_train_subset_to_s3 put_raw_test_data_to_s3
  # run feature extraction and save to features bucket
  make preprocess_raw_train_subset preprocess_raw_test_data
```

#### Third way

```bash
  # put extracted features from full dataset version to s3
  make put_processed_data_to_s3
```

### Traning models

Training models is implemented as individual prefect deployment which operates with previously extracted features. It's assumed that features are stored in separate s3 bucket. There are three pipelines (XGBClassifier, RandomForestClassifier and KNearestNeighborsClassifier):

```bash
  make run_xgboost_training
  make run_random_forest_training
  make run_kneighbors_neighbors_classifier_training
```

Each of runs saves input data, tunes model hyperparameters with [optuna](https://optuna.org/), logs experiment parameters (e.g., metrics, model parameters) using [MLFlow](https://mlflow.org/). It should be noted that only best model's weights is saved to s3 within one experiment.

### Model registry

After training stage two best models by validation G-Mean score are promoted to the registry to the `Production` stage. This stage is implemented as individual prefect deployment as well:

```bash
  make register_best_model make copies
```

Checkpoints of two best registered models are downloaded from s3 and copied into [folder with deployment-ready models](src/deploy/triton_models/).

### Models deployment

Models are deployed using [Triton Inference Server](https://github.com/triton-inference-server/server):

> Triton Inference Server is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more.

Triton is like "advanced" Flask - it supports HTTP/GRPC requests and produces service health metrics which can be collected with [Prometheus](https://prometheus.io/). Another important features are efficient GPU usage, horizontal scaling for different parts of large models, isolated conda environment for each of models and many more. For this project Triton is probably overkill, but it supports all features I needed here, so I decided to use it.

Triton is not intended for use with prefect and provides all necessary tools both for batch and streaming inference, this is the main reason why I didn't make prefect deployment for this stage, not because I can't :) I implemented batch inference, by the words.

#### Inference description

Inference was divided into three stages and implemented with five "models" (each folder with `config.pbtxt` file is called a model in Triton, `config.pbtxt` describes model configuration - batch size, number of model instances, inputs/outputs types and names, model backend and other parameters):

- [pre_processor_1](src/deploy/triton_models/pre_processor_1) extracts batch of features from input audios and passes vectorized features to [predictor_1](src/deploy/triton_models/predictor_1/)
- [predictor_1](src/deploy/triton_models/predictor_1) outputs batch of integer predictions (numbers of classes) for vectorized features batch, checks data drift, and logs results to PostgreSQL database. If data drift occurred, batch of features is passed to [predictor_2](src/deploy/triton_models/predictor_2). This is sort of mock to perform real monitoring, second predictor is not better that first indeed
- [predictor_2](src/deploy/triton_models/predictor_2/) another model to make predictions for batch of vectorized features. Same as first predictor (model kind might differ) except monitoring, this model is outputs predictions
- [post_processor_1](src/deploy/triton_models/post_processor_1/) decodes integer predictions into human-readable music genres names
- [ensemble_1](src/deploy/triton_models/ensemble_1) is ensemble scheduler which orchestrates pre_processor, predictor and post_processor

#### Preparing models' environment

As mentioned above, Triton supports individual environment for each of models, but here environment is shared between models, just to demonstrate this feature. Triton accepts conda environment saved with [conda-pack](https://conda.github.io/conda-pack/) and specified as environment variable inside config file (e.g., [config for ensemble](src/deploy/triton_models/ensemble_1/config.pbtxt)):

```protobuf
  parameters: [
    {
      key: "EXECUTION_ENV_PATH",
      value: {string_value: "/models/conda-pack.tar.gz"}
    }
  ]
```

For reproducibility building environment is implemented in [Dockerfile.condapack](src/deploy/conda-pack/Dockerfile.condapack) and locally can be done as follows:

```bash
  make build_conda_pack
```

Building conda environment is also implemented as [continuous deployment pipeline](https://github.com/ayeffkay/mlops_zoomcamp_project/actions), it outputs models folder with conda-pack archive to use as-is with Triton. If artifact is expired, re-run the job `build_conda_pack`.

#### Running inference

To launch Triton Server, go inside `audioprocessor_server` container and start it:

```bash
  docker exec -it audioprocessor_server bash
  # prepare stage: create buckets shared between client and server for raw data and extracted features
  make make_buckets
  # running web-service
  make start_triton_server
```

When service is ready you will see the following message:
> I0819 16:30:44.071854 62 grpc_server.cc:4819] Started GRPCInferenceService at 0.0.0.0:8001
>
> I0819 16:30:44.072438 62 http_server.cc:3477] Started HTTPService at 0.0.0.0:8000
>
> I0819 16:30:44.114924 62 http_server.cc:184] Started Metrics Service at 0.0.0.0:8002

After that go inside client service and run inference on test data:

```bash
  docker exec -it audioprocessor_client bash
  # login into prefect
  make run_prefect
  make run_client_eval
  make run_client_random

```

The data is chosen so that the drift must occur, so it is the expected behavior that models performs bad.

### Models monitoring

[Models monitoring](src/monitoring/monitor.py) is implemented using [Evidently AI](https://www.evidentlyai.com/), [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) inside [predictor_1](src/deploy/triton_models/predictor_1/1/model.py). PostgreSQL database and Prometheus are specified as data sources for Grafana in [configs](configs), dashboard config is located [here](dashboards/model_metrics.json). To see grafana dashboard, check `localhost:3000`  (default username is `admin`, password is similar).

As mentioned above, when first model makes prediction for batch, evidently calculates data drift and saves result to PostgreSQL database. If data drift is detected, the prediction is made by the second model. Also default Grafana alerting was set - it sends notifications inside Grafana if batch data drift occurred.

## Implemented features: summary

- [x] Problem description: detailed description provided
- [x] Cloud: the project uses localstack, docker images for required services, collected with docker-compose
- [x] Experiments tracking and model registry: both experiment tracking and model registry are used
- [x] Worfkow orchestration: workflow is fully deployed but requires some manual actions due to Triton Inference Server nature
- [x] Model deployment: special tools for model deployment used (Triton Inference Server and Triton Inference Client for requests)
- [x] Model monitoring: comprehensive model monitoring that runs a conditional workflow (switching to different model if data drift occurs) and generates dashboards for service health and data drift metrics
- [x] Reproducibility: instructions are clear (I hope so:)), it's easy to run the code (I hope so too :)) and it must work, I tested it
- [ ] Best practices:
  - [x] There are [unit tests](tests/test_audios.py)
  - [ ] There is an integration test [?] there is no separate integration test, I just run all the code inside docker containers and it worked. I don't understand what else is expected after unit tests.
  - [x] Linters (flake8, pylint) and code formatter (black) are used
  - [x] There's a lot of makefiles ([makefile for project](Makefile), [makefiles for services](makefiles))
  - [x] There are [pre-commit hooks](.pre-commit-config.yaml) for quality checks
  - [x] There's a [CI/CD pipeline](.github/workflows/test.yaml): `build_dev_image` builds and pushes dev image to dockerhub (CD), `run_unit_tests` runs unit tests and computes code coverage from builded image (CI), `build_conda_pack` builds conda environment for Triton Inference server and returns archive as artifact (CD)
