terraform {
  backend "s3" {
    key            = "forecast-sinca-aws/infra.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
