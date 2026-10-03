data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# --- Rol de la task (lo usa el contenedor) -----------------------------------
# Acceso acotado al prefijo de tracking S3 y logs en su log group propio.

resource "aws_iam_role" "mlflow_task" {
  name               = "sinca-mlflow-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "mlflow_task_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.mlflow.arn,
      "${aws_s3_bucket.mlflow.arn}/_mlflow",
      "${aws_s3_bucket.mlflow.arn}/_mlflow/*",
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.mlflow.arn}:*"]
  }
}

resource "aws_iam_role_policy" "mlflow_task_inline" {
  name   = "sinca-mlflow-task-policy"
  role   = aws_iam_role.mlflow_task.id
  policy = data.aws_iam_policy_document.mlflow_task_policy.json
}

# --- Rol de ejecución de la task (lo usa el agente ECS) ----------------------
# Pull de la imagen ECR + escritura de logs. No accede a S3.

resource "aws_iam_role" "mlflow_execution" {
  name               = "sinca-mlflow-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "mlflow_execution_policy" {
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
    resources = [aws_ecr_repository.mlflow.arn]
  }

  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.mlflow.arn}:*"]
  }
}

resource "aws_iam_role_policy" "mlflow_execution_inline" {
  name   = "sinca-mlflow-execution-policy"
  role   = aws_iam_role.mlflow_execution.id
  policy = data.aws_iam_policy_document.mlflow_execution_policy.json
}
