resource "aws_ecr_repository" "mlflow" {
  name                 = "sinca-mlflow"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
