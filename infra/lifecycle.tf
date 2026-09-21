resource "aws_sagemaker_studio_lifecycle_config" "on_start" {
  studio_lifecycle_config_name     = "sinca-on-start"
  studio_lifecycle_config_app_type = "JupyterLab"
  studio_lifecycle_config_content  = base64encode(file("${path.module}/scripts/on_start.sh"))
}
