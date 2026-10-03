terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  # `null` cuando no se pasa profile (ej. GitHub Actions con credenciales OIDC
  # por variables de entorno). Localmente siempre se pasa el named profile real.
  profile = var.aws_profile != "" ? var.aws_profile : null
  region  = "us-east-1"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}
