# app/lambda_function.py

import json
from .model_service import predict_single


def handler(event, context):
    try:
        print("Lambda handler invoked.")
        print(f"Incoming event: {json.dumps(event)}")

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
        print(f"Parsed body: {raw_data}")

        prediction = predict_single(raw_data)
        print(f"Prediction: {prediction}")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"prediction": prediction}),
        }
    except Exception as e:
        print(f"Error in handler: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
