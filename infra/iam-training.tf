data "aws_iam_policy_document" "training_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      "${aws_s3_bucket.data.arn}/validated",
      "${aws_s3_bucket.data.arn}/validated/*",
    ]
  }

  statement {
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
