resource "aws_ecr_repository" "train" {
  name                 = "sinca-train"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
