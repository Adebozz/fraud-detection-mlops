terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # For a portfolio project local state is fine; for teams, use S3 + DynamoDB locking.
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "fraud-detection-mlops"
      ManagedBy = "terraform"
    }
  }
}
