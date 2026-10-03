locals {
  github_repo = "desareca/forecast-sinca-aws"
  oidc_host   = "token.actions.githubusercontent.com"
}

# --- Trust policies ----------------------------------------------------------

# Rol de infra: cualquier ref del repo (el plan corre en PRs). El apply queda
# gateado a main en el workflow (`if: github.ref == 'refs/heads/main'`).
data "aws_iam_policy_document" "github_infra_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "${local.oidc_host}:sub"
      values   = ["repo:${local.github_repo}:*"]
    }
  }
}

# Rol de pipeline: solo `refs/heads/main` del repo.
data "aws_iam_policy_document" "github_pipeline_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = ["repo:${local.github_repo}:ref:refs/heads/main"]
    }
  }
}

# --- Rol OIDC de infraestructura (Terraform plan/apply) ----------------------

resource "aws_iam_role" "github_infra" {
  name               = "sinca-github-infra-role"
  assume_role_policy = data.aws_iam_policy_document.github_infra_assume.json
}

data "aws_iam_policy_document" "github_infra_policy" {
  # State remoto (bucket `<account-id>-tfstate` + tabla de locking).
  statement {
    sid    = "TerraformState"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:GetBucketLocation",
    ]
    resources = [
      "arn:aws:s3:::*-tfstate",
      "arn:aws:s3:::*-tfstate/*",
    ]
  }

  statement {
    sid    = "TerraformStateLock"
    effect = "Allow"
    actions = [
      "dynamodb:DescribeTable",
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
    ]
    resources = [
      "arn:aws:dynamodb:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/terraform-locks",
    ]
  }

  statement {
    sid    = "ListBuckets"
    effect = "Allow"
    actions = [
      "s3:ListAllMyBuckets",
    ]
    resources = ["*"]
  }

  # Buckets gestionados por este Terraform (S3 completo, acotado a ellos).
  statement {
    sid    = "ProjectBuckets"
    effect = "Allow"
    actions = [
      "s3:*",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
      aws_s3_bucket.mlflow.arn,
      "${aws_s3_bucket.mlflow.arn}/*",
    ]
  }

  statement {
    sid    = "Iam"
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:GetRole",
      "iam:ListRoles",
      "iam:UpdateRole",
      "iam:UpdateAssumeRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:ListRoleTags",
      "iam:PutRolePolicy",
      "iam:GetRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:ListRolePolicies",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:ListAttachedRolePolicies",
      "iam:CreatePolicy",
      "iam:GetPolicy",
      "iam:DeletePolicy",
      "iam:ListPolicies",
      "iam:CreatePolicyVersion",
      "iam:GetPolicyVersion",
      "iam:DeletePolicyVersion",
      "iam:ListPolicyVersions",
      "iam:CreateOpenIDConnectProvider",
      "iam:DeleteOpenIDConnectProvider",
      "iam:GetOpenIDConnectProvider",
      "iam:UpdateOpenIDConnectProviderThumbprint",
      "iam:ListOpenIDConnectProviders",
      "iam:AddClientIDToOpenIDConnectProvider",
      "iam:RemoveClientIDFromOpenIDConnectProvider",
      "iam:ListInstanceProfilesForRole",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "PassRole"
    effect = "Allow"
    actions = [
      "iam:PassRole",
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values = [
        "ecs-tasks.amazonaws.com",
        "sagemaker.amazonaws.com",
      ]
    }
  }

  statement {
    sid    = "SageMaker"
    effect = "Allow"
    actions = [
      "sagemaker:CreateDomain",
      "sagemaker:DeleteDomain",
      "sagemaker:DescribeDomain",
      "sagemaker:UpdateDomain",
      "sagemaker:ListDomains",
      "sagemaker:CreateUserProfile",
      "sagemaker:DeleteUserProfile",
      "sagemaker:DescribeUserProfile",
      "sagemaker:UpdateUserProfile",
      "sagemaker:CreateSpace",
      "sagemaker:DeleteSpace",
      "sagemaker:DescribeSpace",
      "sagemaker:UpdateSpace",
      "sagemaker:CreateStudioLifecycleConfig",
      "sagemaker:DeleteStudioLifecycleConfig",
      "sagemaker:DescribeStudioLifecycleConfig",
      "sagemaker:UpdateStudioLifecycleConfig",
      "sagemaker:AddTags",
      "sagemaker:DeleteTags",
      "sagemaker:ListTags",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "Ecr"
    effect = "Allow"
    actions = [
      "ecr:CreateRepository",
      "ecr:DeleteRepository",
      "ecr:DescribeRepositories",
      "ecr:ListTagsForResource",
      "ecr:TagResource",
      "ecr:UntagResource",
      "ecr:PutLifecyclePolicy",
      "ecr:GetLifecyclePolicy",
      "ecr:DeleteLifecyclePolicy",
      "ecr:PutImageScanningConfiguration",
      "ecr:PutImageTagMutability",
      "ecr:SetRepositoryPolicy",
      "ecr:GetRepositoryPolicy",
      "ecr:DeleteRepositoryPolicy",
      "ecr:GetAuthorizationToken",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:DescribeImages",
      "ecr:ListImages",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "Ecs"
    effect = "Allow"
    actions = [
      "ecs:CreateCluster",
      "ecs:DeleteCluster",
      "ecs:DescribeClusters",
      "ecs:UpdateCluster",
      "ecs:RegisterTaskDefinition",
      "ecs:DeregisterTaskDefinition",
      "ecs:DescribeTaskDefinition",
      "ecs:ListTaskDefinitions",
      "ecs:TagResource",
      "ecs:UntagResource",
      "ecs:ListTagsForResource",
      "ecs:RunTask",
      "ecs:StopTask",
      "ecs:DescribeTasks",
      "ecs:ListTasks",
      "ecs:CreateService",
      "ecs:UpdateService",
      "ecs:DeleteService",
      "ecs:DescribeServices",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:DeleteLogGroup",
      "logs:DescribeLogGroups",
      "logs:PutRetentionPolicy",
      "logs:DeleteRetentionPolicy",
      "logs:TagLogGroup",
      "logs:UntagLogGroup",
      "logs:ListTagsLogGroup",
      "logs:PutSubscriptionFilter",
      "logs:DeleteSubscriptionFilter",
      "logs:DescribeSubscriptionFilters",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "Ec2Network"
    effect = "Allow"
    actions = [
      "ec2:DescribeVpcs",
      "ec2:DescribeSubnets",
      "ec2:DescribeSecurityGroups",
      "ec2:DescribeAvailabilityZones",
      "ec2:DescribeAccountAttributes",
      "ec2:DescribeTags",
      "ec2:DescribeNetworkInterfaces",
      "ec2:DescribeVpcAttribute",
      "ec2:DescribeRouteTables",
      "ec2:DescribeInternetGateways",
      "ec2:CreateSecurityGroup",
      "ec2:DeleteSecurityGroup",
      "ec2:AuthorizeSecurityGroupIngress",
      "ec2:RevokeSecurityGroupIngress",
      "ec2:AuthorizeSecurityGroupEgress",
      "ec2:RevokeSecurityGroupEgress",
      "ec2:CreateTags",
      "ec2:DeleteTags",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "StsIdentity"
    effect    = "Allow"
    actions   = ["sts:GetCallerIdentity"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_infra_inline" {
  name   = "sinca-github-infra-policy"
  role   = aws_iam_role.github_infra.id
  policy = data.aws_iam_policy_document.github_infra_policy.json
}

# --- Rol OIDC de pipeline (modelo/artefactos) --------------------------------

resource "aws_iam_role" "github_pipeline" {
  name               = "sinca-github-pipeline-role"
  assume_role_policy = data.aws_iam_policy_document.github_pipeline_assume.json
}

data "aws_iam_policy_document" "github_pipeline_policy" {
  statement {
    sid    = "DataAndArtifacts"
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
      "${aws_s3_bucket.mlflow.arn}/model-artifacts",
      "${aws_s3_bucket.mlflow.arn}/model-artifacts/*",
      "${aws_s3_bucket.mlflow.arn}/_mlflow",
      "${aws_s3_bucket.mlflow.arn}/_mlflow/*",
    ]
  }

  statement {
    sid    = "SageMakerPipeline"
    effect = "Allow"
    actions = [
      "sagemaker:CreatePipeline",
      "sagemaker:UpdatePipeline",
      "sagemaker:DeletePipeline",
      "sagemaker:DescribePipeline",
      "sagemaker:GetPipelineDefinition",
      "sagemaker:ListPipelines",
      "sagemaker:AddTags",
      "sagemaker:DeleteTags",
      "sagemaker:ListTags",
    ]
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

resource "aws_iam_role_policy" "github_pipeline_inline" {
  name   = "sinca-github-pipeline-policy"
  role   = aws_iam_role.github_pipeline.id
  policy = data.aws_iam_policy_document.github_pipeline_policy.json
}
