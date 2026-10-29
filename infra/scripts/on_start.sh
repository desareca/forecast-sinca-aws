#!/bin/bash
# Lifecycle config del Space `sinca-dev` (SageMaker Studio).
#
# Corre al prender el Space y reinstala lo efimero que no vive en EFS.
#
# Regla: un paso de CONVENIENCIA nunca puede dejar el Space sin arrancar. Solo
# un REQUISITO real deja abortar el arranque, y hoy no hay ninguno (el app
# arranca sin ninguna CLI: el repo y los venv ya estan en EFS). Todo lo de
# conveniencia va por `_try`: loguea `WARN (no bloqueante)` y sigue.
#
# Diagnostico: toda la salida queda en ~/.on_start.log (EFS), asi se puede leer
# desde la terminal de JupyterLab aunque el app haya arrancado a medias.

set -uo pipefail   # sin -e: cada fallo se maneja explicitamente, paso por paso

REPO_DIR="/home/sagemaker-user/forecast-sinca-aws"
# El LCC puede correr como root (~ = /root); el log tiene que quedar en el home
# persistente del usuario del Space, que es donde se lee.
LOG_FILE="/home/sagemaker-user/.on_start.log"

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

# Requisito del Space: sin esto el entorno no sirve -> aborta el arranque.
_fail() {
  log "ERROR: $*"
  log "=== on_start END (failed) ==="
  exit 1
}

# Conveniencia: se loguea el fallo y se sigue. No bloquea el arranque del Space.
_try() {
  local desc="$1"; shift
  local rc=0
  "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then
    log "OK: $desc"
  else
    log "WARN (no bloqueante): '$desc' fallo (exit $rc) — el Space arranca igual"
  fi
}

install_aws_cli() {
  command -v aws >/dev/null 2>&1 && return 0
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip &&
    unzip -q -o /tmp/awscliv2.zip -d /tmp &&
    /tmp/aws/install
}

install_terraform() {
  command -v terraform >/dev/null 2>&1 && return 0
  apt-get install -y gnupg software-properties-common &&
    wget -qO- https://apt.releases.hashicorp.com/gpg |
    gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg &&
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/hashicorp.list &&
    apt-get update &&
    apt-get install -y terraform
}

clone_repo() {
  [ -d "$REPO_DIR/.git" ] && return 0
  # Por SSH: el Space no tiene credenciales (corre con `sinca-dev-role`); la key
  # ed25519 vive en ~/.ssh y persiste en EFS (se configura una sola vez, ver
  # README -> "Entorno de desarrollo -> Git desde el Space"). `accept-new` deja
  # el host key de GitHub en known_hosts en el primer clone (sin prompt de TTY).
  # Si la key todavia no existe, este paso queda en WARN y se clona a mano.
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" \
    git clone git@github.com:desareca/forecast-sinca-aws.git "$REPO_DIR"
}

main() {
  log "=== on_start BEGIN (user=$(id -un), repo=$REPO_DIR) ==="

  _try "AWS CLI v2" install_aws_cli
  _try "Terraform" install_terraform
  _try "repo clonado por SSH" clone_repo

  log "=== on_start END (ok) ==="
}

main 2>&1 | tee -a "$LOG_FILE"
