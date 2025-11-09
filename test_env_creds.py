#!/usr/bin/env python3

import sys
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name, region_name="us-east-1"):
    """
    Retrieve a secret string from AWS Secrets Manager.

    Args:
        secret_name (str): The name of the secret.
        region_name (str): AWS region name.

    Returns:
        str: The secret value as a string.

    Raises:
        Exception: If retrieval fails.
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # See: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        print(f"Error getting secret {secret_name}: {e}")
        raise e

    # SecretString contains the stored secret; if not, it might be binary
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
    else:
        secret = get_secret_value_response['SecretBinary']
    return secret

def main():
    region = "us-east-1"
    secret_names = [
        "ai_agent_aws_agentcore_01/prod/slack_webhook_url",
        "ai_agent_aws_agentcore_01/prod/bedrock_agentcore_memory"
    ]
    for secret_name in secret_names:
        try:
            secret = get_secret(secret_name, region)
            print(f"Secret value for '{secret_name}' retrieved successfully.")
            # For demonstration, print only a summary, not the secret.
        except Exception as e:
            print(f"Failed to retrieve secret: {secret_name}")
            sys.exit(1)

if __name__ == "__main__":
    main()

