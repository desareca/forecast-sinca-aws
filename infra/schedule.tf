# Rol que asume EventBridge Scheduler para disparar `ecs:RunTask`.

data "aws_iam_policy_document" "scraper_scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scraper_scheduler" {
  name               = "sinca-scraper-scheduler-role"
  assume_role_policy = data.aws_iam_policy_document.scraper_scheduler_assume.json
}

data "aws_iam_policy_document" "scraper_scheduler_policy" {
  statement {
    effect    = "Allow"
    actions   = ["ecs:RunTask"]
    resources = [aws_ecs_task_definition.scraper.arn_without_revision]
  }

  statement {
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.scraper_task.arn, aws_iam_role.scraper_execution.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "scraper_scheduler_inline" {
  name   = "sinca-scraper-scheduler-policy"
  role   = aws_iam_role.scraper_scheduler.id
  policy = data.aws_iam_policy_document.scraper_scheduler_policy.json
}

# Regla diaria (~01:00 America/Santiago) que dispara la task del scraper
# (modo incremental por defecto).
resource "aws_scheduler_schedule" "scraper_daily" {
  name       = "sinca-scraper-daily"
  group_name = "default"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = var.scraper_schedule_expression
  schedule_expression_timezone = var.scraper_schedule_timezone

  target {
    arn      = aws_ecs_cluster.scraper.arn
    role_arn = aws_iam_role.scraper_scheduler.arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.scraper.arn
      launch_type         = "FARGATE"

      network_configuration {
        subnets          = data.aws_subnets.default.ids
        security_groups  = [aws_security_group.scraper.id]
        assign_public_ip = true
      }
    }

    retry_policy {
      maximum_event_age_in_seconds = 3600
      maximum_retry_attempts       = 2
    }
  }
}
