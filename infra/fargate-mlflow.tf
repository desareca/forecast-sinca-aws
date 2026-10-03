resource "aws_ecs_cluster" "mlflow" {
  name = "sinca-mlflow"
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/sinca-mlflow"
  retention_in_days = 7
}

# Acceso a la UI de MLflow (puerto 5000). La task se levanta con IP pública
# en el default VPC; restringir `mlflow_allowed_cidr` a la IP del operador.
resource "aws_security_group" "mlflow" {
  name        = "sinca-mlflow"
  description = "Acceso a la UI de MLflow on-demand (puerto 5000)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "UI MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.mlflow_allowed_cidr]
  }

  egress {
    description = "Salida a S3/ECR/CloudWatch"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "mlflow" {
  family                   = "sinca-mlflow"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.mlflow_task_cpu
  memory                   = var.mlflow_task_memory
  execution_role_arn       = aws_iam_role.mlflow_execution.arn
  task_role_arn            = aws_iam_role.mlflow_task.arn

  container_definitions = jsonencode([
    {
      name      = "mlflow"
      image     = "${aws_ecr_repository.mlflow.repository_url}:latest"
      essential = true
      portMappings = [
        {
          containerPort = 5000
          hostPort      = 5000
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "MLFLOW_BUCKET", value = var.mlflow_bucket_name },
        { name = "MLFLOW_PREFIX", value = "_mlflow" },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.mlflow.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "mlflow"
        }
      }
    }
  ])
}
