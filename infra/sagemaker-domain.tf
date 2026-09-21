data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_sagemaker_domain" "studio" {
  domain_name             = var.sagemaker_domain_name
  auth_mode               = "IAM"
  vpc_id                  = data.aws_vpc.default.id
  subnet_ids              = data.aws_subnets.default.ids
  app_network_access_type = "PublicInternetOnly"

  default_user_settings {
    execution_role = aws_iam_role.dev.arn
  }

  default_space_settings {
    execution_role = aws_iam_role.dev.arn
  }

  retention_policy {
    home_efs_file_system = "Retain"
  }
}

resource "aws_sagemaker_user_profile" "admin" {
  domain_id         = aws_sagemaker_domain.studio.id
  user_profile_name = var.user_profile_name

  user_settings {
    execution_role = aws_iam_role.dev.arn
  }
}

resource "aws_sagemaker_space" "dev" {
  domain_id  = aws_sagemaker_domain.studio.id
  space_name = var.space_name

  ownership_settings {
    owner_user_profile_name = aws_sagemaker_user_profile.admin.user_profile_name
  }

  space_sharing_settings {
    sharing_type = "Private"
  }

  space_settings {
    app_type = "JupyterLab"

    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type        = var.space_instance_type
        lifecycle_config_arn = aws_sagemaker_studio_lifecycle_config.on_start.arn
      }
    }
  }
}
