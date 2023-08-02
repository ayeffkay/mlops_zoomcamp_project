quality_checks:
	isort .
	black .
	pylint --recursive=y .
setup:
	pipenv install --dev
	pre-commit install