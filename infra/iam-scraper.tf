# --- Rol de la task del scraper (lo usa el contenedor) -----------------------
# Acceso S3 acotado a `sinca-data/*` y logs en su log group propio.
# Sin permisos IAM ni managed policies amplias. Open-Meteo / SINCA / Nager.Date
# son HTTP público (no requieren IAM).

resource "aws_iam_role" "scraper_task" {
  name               = "sinca-scraper-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "scraper_task_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.scraper.arn}:*"]
  }
}

resource "aws_iam_role_policy" "scraper_task_inline" {
  name   = "sinca-scraper-task-policy"
  role   = aws_iam_role.scraper_task.id
  policy = data.aws_iam_policy_document.scraper_task_policy.json
}

# --- Rol de ejecución de la task (lo usa el agente ECS) ----------------------
# Pull de la imagen ECR + escritura de logs. No accede a S3.

resource "aws_iam_role" "scraper_execution" {
  name               = "sinca-scraper-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "scraper_execution_policy" {
  statement {
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = [aws_ecr_repository.scraper.arn]
  }

  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.scraper.arn}:*"]
  }
}

resource "aws_iam_role_policy" "scraper_execution_inline" {
  name   = "sinca-scraper-execution-policy"
  role   = aws_iam_role.scraper_execution.id
  policy = data.aws_iam_policy_document.scraper_execution_policy.json
}
