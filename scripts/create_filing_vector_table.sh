#!/usr/bin/env bash
set -euo pipefail

TABLE_NAME="${FILING_VECTOR_TABLE_NAME:-agentcore_filing_vectors}"
AWS_REGION="${AWS_REGION:-us-west-2}"

if aws dynamodb describe-table --table-name "${TABLE_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "Table ${TABLE_NAME} already exists"
else
  echo "Creating DynamoDB table ${TABLE_NAME}"
  aws dynamodb create-table \
    --table-name "${TABLE_NAME}" \
    --attribute-definitions AttributeName=session_id,AttributeType=S AttributeName=chunk_id,AttributeType=S \
    --key-schema AttributeName=session_id,KeyType=HASH AttributeName=chunk_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region "${AWS_REGION}" >/dev/null
  aws dynamodb wait table-exists --table-name "${TABLE_NAME}" --region "${AWS_REGION}"
fi

echo "Enabling TTL on expires_at"
aws dynamodb update-time-to-live \
  --table-name "${TABLE_NAME}" \
  --time-to-live-specification "Enabled=true,AttributeName=expires_at" \
  --region "${AWS_REGION}" >/dev/null 2>&1 || echo "TTL already enabled"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
cat <<EOF

Grant the app role only these actions:
{
  "Effect": "Allow",
  "Action": ["dynamodb:BatchWriteItem", "dynamodb:PutItem", "dynamodb:Query"],
  "Resource": "arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${TABLE_NAME}"
}
EOF
