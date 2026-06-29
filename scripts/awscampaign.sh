#!/usr/bin/env bash
set -euo pipefail
: "${S3_BUCKET:?set S3_BUCKET}"
: "${AWS_REGION:?set AWS_REGION}"
cd aws/terraform
terraform init
terraform apply -var="s3_bucket=${S3_BUCKET}" -var="aws_region=${AWS_REGION}"
