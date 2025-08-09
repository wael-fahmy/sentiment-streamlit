import boto3
import json
import os
import logging

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "ai-sentiment-endpoint")

def lambda_handler(event, context):
    try:
        logger.info(f"Received event: {json.dumps(event)}")

        body = json.loads(event["body"])
        text = body.get("text", "")

        if not text:
            logger.warning("No text provided in request.")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'text' in request"})
            }

        logger.info(f"Analyzing text: {text}")

        payload = json.dumps({"inputs": text})
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload
        )

        result = json.loads(response['Body'].read().decode())
        logger.info(f"SageMaker response: {result}")

        return {
            "statusCode": 200,
            "body": json.dumps({"result": result})
        }

    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
