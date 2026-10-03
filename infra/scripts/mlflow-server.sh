#!/usr/bin/env bash
set -euo pipefail

# Operación del servidor MLflow on-demand (task Fargate).
#   up   -> levanta la task y muestra la URL pública de la UI
#   down -> detiene la task (dispara SIGTERM; el entrypoint sube el .sqlite a S3)
#
# Requiere AWS_PROFILE (named profile explícito de la cuenta personal).
# Opcional: AWS_REGION (default us-east-1).

: "${AWS_PROFILE:?Definí AWS_PROFILE (named profile de la cuenta personal)}"
REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${MLFLOW_CLUSTER:-sinca-mlflow}"
TASK_DEF="${MLFLOW_TASK_DEF:-sinca-mlflow}"
SG_NAME="${MLFLOW_SG_NAME:-sinca-mlflow}"

AWS=(aws --profile "$AWS_PROFILE" --region "$REGION")

usage() {
  echo "Uso: $0 {up|down}" >&2
  exit 1
}

default_vpc_id() {
  "${AWS[@]}" ec2 describe-vpcs \
    --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text
}

default_subnets() {
  local vpc_id="$1"
  "${AWS[@]}" ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${vpc_id}" \
    --query 'Subnets[].SubnetId' --output text | tr '\t' ','
}

security_group_id() {
  local vpc_id="$1"
  "${AWS[@]}" ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${vpc_id}" \
    --query 'SecurityGroups[0].GroupId' --output text
}

cmd_up() {
  local vpc_id subnets sg_id task_arn eni public_ip
  vpc_id="$(default_vpc_id)"
  subnets="$(default_subnets "$vpc_id")"
  sg_id="$(security_group_id "$vpc_id")"

  echo "[up] cluster=${CLUSTER} task-def=${TASK_DEF} vpc=${vpc_id} sg=${sg_id}"
  task_arn="$("${AWS[@]}" ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${subnets}],securityGroups=[${sg_id}],assignPublicIp=ENABLED}" \
    --query 'tasks[0].taskArn' --output text)"

  echo "[up] task lanzada: ${task_arn}"
  echo "[up] esperando a RUNNING..."
  "${AWS[@]}" ecs wait tasks-running --cluster "$CLUSTER" --tasks "$task_arn"

  eni="$("${AWS[@]}" ecs describe-tasks --cluster "$CLUSTER" --tasks "$task_arn" \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)"
  public_ip="$("${AWS[@]}" ec2 describe-network-interfaces --network-interface-ids "$eni" \
    --query 'NetworkInterfaces[0].Association.PublicIp' --output text)"

  echo "[up] UI MLflow: http://${public_ip}:5000"
}

cmd_down() {
  local tasks task_arn
  tasks="$("${AWS[@]}" ecs list-tasks --cluster "$CLUSTER" --desired-status RUNNING \
    --family "$TASK_DEF" --query 'taskArns[]' --output text)"
  if [ -z "$tasks" ] || [ "$tasks" = "None" ]; then
    echo "[down] no hay tasks corriendo"
    return 0
  fi
  for task_arn in $tasks; do
    echo "[down] deteniendo ${task_arn}"
    "${AWS[@]}" ecs stop-task --cluster "$CLUSTER" --task "$task_arn" \
      --reason "apagado manual via mlflow-server.sh" >/dev/null
  done
  echo "[down] tasks detenidas (el .sqlite se sube a S3 en el SIGTERM)"
}

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  *) usage ;;
esac
