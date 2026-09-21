data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "dev_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
      "${aws_s3_bucket.mlflow.arn}/_mlflow",
      "${aws_s3_bucket.mlflow.arn}/_mlflow/*",
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

  statement {
    effect = "Allow"
    actions = [
      "sagemaker:CreatePresignedDomainUrl",
      "sagemaker:DescribeDomain",
      "sagemaker:ListDomains",
      "sagemaker:DescribeUserProfile",
      "sagemaker:DescribeSpace",
      "sagemaker:CreateApp",
      "sagemaker:DescribeApp",
      "sagemaker:DeleteApp",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role" "dev" {
  name               = "sinca-dev-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy" "dev_inline" {
  name   = "sinca-dev-policy"
  role   = aws_iam_role.dev.id
  policy = data.aws_iam_policy_document.dev_policy.json
}
