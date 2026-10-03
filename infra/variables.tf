variable "aws_profile" {
  description = "AWS named profile de la cuenta personal (setear localmente en terraform.tfvars, no versionado)"
  type        = string
}

variable "data_bucket_name" {
  description = "Nombre del bucket S3 de datos"
  type        = string
  default     = "sinca-data"
}

variable "mlflow_bucket_name" {
  description = "Nombre del bucket S3 de tracking MLflow"
  type        = string
  default     = "sinca-mlflow"
}

variable "sagemaker_domain_name" {
  description = "Nombre del dominio SageMaker Studio"
  type        = string
  default     = "sinca-studio"
}

variable "user_profile_name" {
  description = "Nombre del user profile de Studio"
  type        = string
  default     = "sinca-dev-user"
}

variable "space_name" {
  description = "Nombre del Space de desarrollo"
  type        = string
  default     = "sinca-dev"
}

variable "space_instance_type" {
  description = "Tipo de instancia del Space"
  type        = string
  default     = "ml.t3.large"
}

variable "mlflow_allowed_cidr" {
  description = "CIDR autorizado a acceder a la UI de MLflow (puerto 5000). Restringir a la IP del operador."
  type        = string
  default     = "0.0.0.0/0"
}

variable "mlflow_task_cpu" {
  description = "CPU de la task Fargate de MLflow (unidades de 1/1024 vCPU)"
  type        = string
  default     = "1024"
}

variable "mlflow_task_memory" {
  description = "Memoria (MiB) de la task Fargate de MLflow"
  type        = string
  default     = "2048"
}
