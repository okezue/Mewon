terraform {
  required_providers { aws = { source = "hashicorp/aws", version = ">= 5.0" } }
}
provider "aws" { region = var.aws_region }
resource "aws_s3_bucket" "results" { bucket = var.s3_bucket }
# GPU instance creation is intentionally user-controlled. Supply AMI, key, VPC, subnet, IAM, security group.
# This file is complete enough to own result storage and acts as the safe scaffold for your cluster module.
