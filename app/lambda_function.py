# app/lambda_function.py

import json
from .model_service import predict_single


def handler(event, context):
    try:
        body = event.get("body")

        if body is None:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Request body is required"}),
            }

        if isinstance(body, str):
            body = json.loads(body)

        if not isinstance(body, dict):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Body must be a JSON object"}),
            }

        raw_data = body

        prediction = predict_single(raw_data)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"prediction": prediction}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
