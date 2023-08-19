export PYTHONPATH:=./src
setup:
	pip install pipenv==2023.7.23
	pipenv install && pipenv shell
	pipenv install --dev black==23.7.0 mypy==1.5.0 isort==5.12.0 pylint==2.17.5 flake8==6.1.0 pytest==7.4.0 pytest-cov==4.1.0
pre_commit:
	pipenv install --dev pre-commit==3.3.3
	pre-commit install
unit_tests:
	pytest --cov=./src --cov-report=xml tests/
quality_checks:
	find ./src  -type f -name "*.py" -exec isort {} \;
	find ./src  -type f -name "*.py" -exec black {} \;
	find ./src  -type f -name "*.py" -exec mypy --install-types --ignore-missing-imports \
																--disable-error-code=misc \
																--disable-error-code=unused-coroutine \
																--disable-error-code=has-type \
																--disable-error-code=call-overload \
																--disable-error-code=attr-defined {} \;
	find ./src  -type f -name "*.py" -exec flake8 --ignore=E501,W291 {} \;
	find ./src  -type f -name "*.py" -exec pylint --fail-under 0 --disable=missing-class-docstring \
																--disable=missing-function-docstring \
																--disable=missing-module-docstring \
																--disable=no-value-for-parameter \
																--disable=consider-using-from-import \
																--disable=invalid-name {} \;
build_conda_pack:
	cd src/deploy/conda-pack && docker build -f Dockerfile.condapack --tag conda-pack --output "../triton_models" .
build_all:
	docker compose -f docker-compose.yaml build
up_all:
	docker compose -f docker-compose.yaml up
build_and_up:
	docker compose -f docker-compose.yaml up --build
