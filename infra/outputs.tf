output "data_bucket_arn" {
  value = aws_s3_bucket.data.arn
}

output "mlflow_bucket_arn" {
  value = aws_s3_bucket.mlflow.arn
}

output "sagemaker_domain_id" {
  value = aws_sagemaker_domain.studio.id
}

output "sagemaker_space_name" {
  value = aws_sagemaker_space.dev.space_name
}

output "dev_role_arn" {
  value = aws_iam_role.dev.arn
}

output "training_role_arn" {
  value = aws_iam_role.training.arn
}

output "mlflow_ecr_repository_url" {
  value = aws_ecr_repository.mlflow.repository_url
}

output "mlflow_ecr_repository_arn" {
  value = aws_ecr_repository.mlflow.arn
}

output "mlflow_ecs_cluster_name" {
  value = aws_ecs_cluster.mlflow.name
}

output "mlflow_task_definition_arn" {
  value = aws_ecs_task_definition.mlflow.arn
}

output "mlflow_security_group_id" {
  value = aws_security_group.mlflow.id
}

output "mlflow_task_role_arn" {
  value = aws_iam_role.mlflow_task.arn
}

output "mlflow_execution_role_arn" {
  value = aws_iam_role.mlflow_execution.arn
}

output "github_oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.github.arn
}

output "github_infra_role_arn" {
  value = aws_iam_role.github_infra.arn
}

output "github_pipeline_role_arn" {
  value = aws_iam_role.github_pipeline.arn
}

output "scraper_ecr_repository_url" {
  value = aws_ecr_repository.scraper.repository_url
}

output "scraper_ecr_repository_arn" {
  value = aws_ecr_repository.scraper.arn
}

output "scraper_ecs_cluster_name" {
  value = aws_ecs_cluster.scraper.name
}

output "scraper_task_definition_arn" {
  value = aws_ecs_task_definition.scraper.arn
}

output "scraper_security_group_id" {
  value = aws_security_group.scraper.id
}

output "scraper_task_role_arn" {
  value = aws_iam_role.scraper_task.arn
}

output "scraper_execution_role_arn" {
  value = aws_iam_role.scraper_execution.arn
}

output "scraper_schedule_arn" {
  value = aws_scheduler_schedule.scraper_daily.arn
}
