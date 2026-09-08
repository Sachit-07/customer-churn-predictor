import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


import json
def get_business_advice(customer_info: dict, prediction: str, confidence: float) -> dict:
    prompt = f"""You are an AI Business Advisor inside a customer-churn dashboard.
A machine learning model has ALREADY predicted the churn outcome below — do not
re-predict or second-guess it, only explain it.

Customer profile: {json.dumps(customer_info, indent=2)}
ML model prediction: {prediction}
Confidence: {confidence:.0%}

Respond ONLY with a JSON object (no markdown fences, no extra text) with exactly these keys:
{{
  "explanation": "2-3 sentences explaining why this customer got this prediction",
  "risk_factors": ["short risk factor 1", "short risk factor 2"],
  "retention_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}
Ground every point in the specific customer details given above."""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    raw_text = response.text.strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"explanation": raw_text, "risk_factors": [], "retention_suggestions": []}


