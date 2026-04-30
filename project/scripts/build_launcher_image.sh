#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${1:-research-platform-launcher:latest}"

docker build \
  -t "${IMAGE_NAME}" \
  -f "${PROJECT_DIR}/docker/launcher.Dockerfile" \
  "${PROJECT_DIR}"

echo "Built ${IMAGE_NAME}"
