if aws s3 --endpoint-url=http://localstack:4566 ls mlflow 2>&1 | grep -q 'NoSuchBucket';
then
    aws --endpoint-url=http://localstack:4566 s3 mb s3://mlflow;
fi
mlflow ui --backend-store-uri postgresql+psycopg2://postgres:postgres@db:5432/mlflow_db \
          --default-artifact-root s3://mlflow \
          --host 0.0.0.0:5001 --no-serve-artifacts
