# --- Rol de training (SageMaker Training Job) --------------------------------
# Finalizado en Fase 3 (change `baseline`). Mínimo privilegio, acotado a:
#   - S3 read solo en `sinca-data/validated/*` (datos de entrada).
#   - S3 r/w en `sinca-mlflow/_mlflow/*` (tracking: sqlite + mlruns + código) y
#     `sinca-mlflow/model-artifacts/*` (modelos versionados).
#   - ECR pull acotado al repo `sinca-train` (SageMaker baja la imagen custom).
#   - Logs en su log group propio `/aws/sagemaker/*`.
# Sin permisos IAM ni managed policies amplias.

data "aws_iam_policy_document" "training_policy" {
  statement {
    sid    = "ReadValidatedData"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/validated",
      "${aws_s3_bucket.data.arn}/validated/*",
    ]
  }

  statement {
    sid    = "ReadWriteTracking"
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
    sid    = "WriteModelArtifacts"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      "${aws_s3_bucket.mlflow.arn}/model-artifacts",
      "${aws_s3_bucket.mlflow.arn}/model-artifacts/*",
    ]
  }

  statement {
    sid    = "PullTrainingImage"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.train.arn]
  }

  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "Logs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = [
      "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*",
    ]
  }
}

resource "aws_iam_role" "training" {
  name               = "sinca-training-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy" "training_inline" {
  name   = "sinca-training-policy"
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.training_policy.json
}
