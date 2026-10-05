resource "aws_ecs_cluster" "scraper" {
  name = "sinca-scraper"
}

resource "aws_cloudwatch_log_group" "scraper" {
  name              = "/ecs/sinca-scraper"
  retention_in_days = 7
}

# La task solo necesita salida a internet (SINCA / Open-Meteo / Nager.Date) y
# a S3/ECR/CloudWatch. Sin ingress.
resource "aws_security_group" "scraper" {
  name        = "sinca-scraper"
  description = "Egress del scraper on-demand (sin ingress)"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Salida a SINCA / Open-Meteo / Nager.Date / S3 / ECR / CloudWatch"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "scraper" {
  family                   = "sinca-scraper"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.scraper_task_cpu
  memory                   = var.scraper_task_memory
  execution_role_arn       = aws_iam_role.scraper_execution.arn
  task_role_arn            = aws_iam_role.scraper_task.arn

  container_definitions = jsonencode([
    {
      name      = "scraper"
      image     = "${aws_ecr_repository.scraper.repository_url}:latest"
      essential = true
      environment = [
        { name = "DATA_BUCKET", value = var.data_bucket_name },
        { name = "AWS_DEFAULT_REGION", value = data.aws_region.current.name },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.scraper.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "scraper"
        }
      }
    }
  ])
}
