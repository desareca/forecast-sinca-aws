# --- Orquestación de deep-koopman (D18, D19) -------------------------------------
# Máquina de estados Standard en serie, Lambda de decisión (sin reglas), avisos SNS por email y
# regla EventBridge de fallos. Mínimo privilegio por ARN/prefijo. El rol de training no cambia.

locals {
  koop_nombre      = "sinca-deep-koopman"
  koop_orq_prefijo = "model-artifacts/orquestacion/deep-koopman"
  koop_cuenta      = data.aws_caller_identity.current.account_id
  koop_region      = data.aws_region.current.name
}

# --- Lambda de decisión ------------------------------------------------------------

data "archive_file" "koop_decision" {
  type             = "zip"
  source_file      = "${path.module}/lambda/decision_rama/handler.py"
  output_path      = "${path.module}/.build/decision_rama.zip"
  output_file_mode = "0666"
}

resource "aws_cloudwatch_log_group" "koop_decision" {
  name              = "/aws/lambda/${local.koop_nombre}-decision"
  retention_in_days = 30
}

data "aws_iam_policy_document" "koop_lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "koop_decision" {
  name               = "${local.koop_nombre}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.koop_lambda_assume.json
}

data "aws_iam_policy_document" "koop_decision" {
  statement {
    sid       = "LeerDecisionesYMarcadores"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.mlflow.arn}/${local.koop_orq_prefijo}/*"]
  }

  # ListBucket acotado al prefijo: un marcador ausente da 404 (no 403).
  statement {
    sid       = "ListarPrefijoOrquestacion"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.mlflow.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${local.koop_orq_prefijo}/*"]
    }
  }

  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.koop_decision.arn}:*"]
  }
}

resource "aws_iam_role_policy" "koop_decision" {
  name   = "${local.koop_nombre}-lambda-policy"
  role   = aws_iam_role.koop_decision.id
  policy = data.aws_iam_policy_document.koop_decision.json
}

resource "aws_lambda_function" "koop_decision" {
  function_name    = "${local.koop_nombre}-decision"
  role             = aws_iam_role.koop_decision.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.koop_decision.output_path
  source_code_hash = data.archive_file.koop_decision.output_base64sha256
  timeout          = 30
  memory_size      = 128
  depends_on       = [aws_cloudwatch_log_group.koop_decision]
}

# --- Avisos (SNS) ------------------------------------------------------------------

resource "aws_sns_topic" "koop_avisos" {
  name = "${local.koop_nombre}-avisos"
}

resource "aws_sns_topic_subscription" "koop_email" {
  topic_arn = aws_sns_topic.koop_avisos.arn
  protocol  = "email"
  endpoint  = var.operator_email
}

# Publica EventBridge (fallos de la máquina). Step Functions publica con su rol (misma cuenta).
data "aws_iam_policy_document" "koop_avisos_topic" {
  statement {
    sid       = "EventBridgePublica"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.koop_avisos.arn]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [aws_cloudwatch_event_rule.koop_fallos.arn]
    }
  }
}

resource "aws_sns_topic_policy" "koop_avisos" {
  arn    = aws_sns_topic.koop_avisos.arn
  policy = data.aws_iam_policy_document.koop_avisos_topic.json
}

# --- Fallos de ejecución (EventBridge -> SNS) ----------------------------------------

resource "aws_cloudwatch_event_rule" "koop_fallos" {
  name        = "${local.koop_nombre}-fallos"
  description = "Ejecuciones de ${local.koop_nombre} que terminan en FAILED, TIMED_OUT o ABORTED"
  event_pattern = jsonencode({
    source        = ["aws.states"]
    "detail-type" = ["Step Functions Execution Status Change"]
    detail = {
      status          = ["FAILED", "TIMED_OUT", "ABORTED"]
      stateMachineArn = ["arn:aws:states:${local.koop_region}:${local.koop_cuenta}:stateMachine:${local.koop_nombre}"]
    }
  })
}

resource "aws_cloudwatch_event_target" "koop_fallos_sns" {
  rule = aws_cloudwatch_event_rule.koop_fallos.name
  arn  = aws_sns_topic.koop_avisos.arn
  input_transformer {
    input_paths = {
      nombre = "$.detail.name"
      estado = "$.detail.status"
    }
    input_template = "\"deep-koopman: la ejecucion <nombre> termino en <estado>. Revisar el historial en Step Functions.\""
  }
}

# --- Rol de Step Functions -----------------------------------------------------------

data "aws_iam_policy_document" "koop_sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "koop_sfn" {
  name               = "${local.koop_nombre}-sfn-role"
  assume_role_policy = data.aws_iam_policy_document.koop_sfn_assume.json
}

data "aws_iam_policy_document" "koop_sfn" {
  statement {
    sid    = "TrainingJobsKoopman"
    effect = "Allow"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:AddTags",
    ]
    resources = ["arn:aws:sagemaker:${local.koop_region}:${local.koop_cuenta}:training-job/sinca-koop-*"]
  }

  statement {
    sid       = "PasarRolDeTraining"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.training.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # Regla administrada que usa la integración síncrona (.sync) con SageMaker.
  statement {
    sid       = "ReglaSync"
    effect    = "Allow"
    actions   = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
    resources = ["arn:aws:events:${local.koop_region}:${local.koop_cuenta}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule"]
  }

  statement {
    sid       = "InvocarDecision"
    effect    = "Allow"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.koop_decision.arn, "${aws_lambda_function.koop_decision.arn}:*"]
  }

  statement {
    sid       = "Avisos"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.koop_avisos.arn]
  }
}

resource "aws_iam_role_policy" "koop_sfn" {
  name   = "${local.koop_nombre}-sfn-policy"
  role   = aws_iam_role.koop_sfn.id
  policy = data.aws_iam_policy_document.koop_sfn.json
}

# --- Máquina de estados --------------------------------------------------------------

resource "aws_sfn_state_machine" "koop" {
  name     = local.koop_nombre
  role_arn = aws_iam_role.koop_sfn.arn
  type     = "STANDARD"
  definition = templatefile("${path.module}/stepfunctions/deep-koopman.asl.json", {
    lambda_arn = aws_lambda_function.koop_decision.arn
    topic_arn  = aws_sns_topic.koop_avisos.arn
  })
}

output "koop_state_machine_arn" {
  description = "ARN de la máquina de estados de deep-koopman"
  value       = aws_sfn_state_machine.koop.arn
}

output "koop_avisos_topic_arn" {
  description = "Tópico SNS de avisos de deep-koopman"
  value       = aws_sns_topic.koop_avisos.arn
}
