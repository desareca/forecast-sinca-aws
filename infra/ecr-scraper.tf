resource "aws_ecr_repository" "scraper" {
  name                 = "sinca-scraper"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
