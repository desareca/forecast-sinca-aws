#!/bin/bash
set -euo pipefail

# Reinstala lo efímero que no persiste en EFS (corre al prender el Space).

# Node.js + Open Code CLI
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

if ! command -v opencode >/dev/null 2>&1; then
  npm install -g @opencode-ai/opencode
fi

# AWS CLI
if ! command -v aws >/dev/null 2>&1; then
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install
fi

# Terraform
if ! command -v terraform >/dev/null 2>&1; then
  apt-get install -y gnupg software-properties-common
  wget -qO- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" > /etc/apt/sources.list.d/hashicorp.list
  apt-get update
  apt-get install -y terraform
fi

# Clonar el repo (persistido en EFS; solo si no existe).
# Por SSH: el Space no tiene credenciales (corre con `sinca-dev-role`), la key
# ed25519 vive en ~/.ssh y persiste en EFS (se configura una sola vez, ver
# README -> "Entorno de desarrollo -> Git desde el Space"). `accept-new` deja
# el host key de GitHub en known_hosts en el primer clone (sin prompt de TTY).
REPO_DIR="/home/sagemaker-user/forecast-sinca-aws"
if [ ! -d "$REPO_DIR/.git" ]; then
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" \
    git clone git@github.com:desareca/forecast-sinca-aws.git "$REPO_DIR" || true
fi
