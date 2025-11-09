#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-agentcore-faq}"
AWS_REGION="${AWS_REGION:-us-west-2}"
IMAGE_TAG="${IMAGE_TAG:-$(cd "${PROJECT_ROOT}" && git rev-parse --short HEAD)}"

ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPOSITORY_URI="${ECR_URI}/${IMAGE_NAME}"

echo "Building Docker image ${REPOSITORY_URI}:${IMAGE_TAG}"
docker build --platform linux/amd64 -t "${REPOSITORY_URI}:${IMAGE_TAG}" "${PROJECT_ROOT}"

echo "Ensuring ECR repository ${IMAGE_NAME} exists"
aws ecr describe-repositories \
  --repository-names "${IMAGE_NAME}" \
  --region "${AWS_REGION}" >/dev/null 2>&1 || \
aws ecr create-repository \
  --repository-name "${IMAGE_NAME}" \
  --image-scanning-configuration scanOnPush=true \
  --region "${AWS_REGION}"

echo "Logging in to ECR ${ECR_URI}"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_URI}"

echo "Pushing image to ${REPOSITORY_URI}:${IMAGE_TAG}"
docker push "${REPOSITORY_URI}:${IMAGE_TAG}"

echo "Image pushed successfully: ${REPOSITORY_URI}:${IMAGE_TAG}"

