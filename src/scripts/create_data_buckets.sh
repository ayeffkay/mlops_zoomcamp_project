if aws s3 --endpoint-url=http://localstack:4566 ls s3://${S3_RAW_DATA_BUCKET} 2>&1 | grep -q 'NoSuchBucket';
then
    aws --endpoint-url=http://localstack:4566 s3 mb s3://${S3_RAW_DATA_BUCKET};
fi
if aws s3 --endpoint-url=http://localstack:4566 ls s3://${S3_FEATURES_BUCKET} 2>&1 | grep -q 'NoSuchBucket';
then
    aws --endpoint-url=http://localstack:4566 s3 mb s3://${S3_FEATURES_BUCKET};
fi
