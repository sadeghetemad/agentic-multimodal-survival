import json
import re

from services.feature_parser_service import parse

# Fix Broken Json
def repair_json(text: str):

    # remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    if text.count('"') % 2 != 0:
        text += '"'

    return text


# Extract Json Block
def extract_json(text: str):

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON found:\n{text}")

    json_str = match.group(0)

    # try direct
    try:
        return json.loads(json_str)
    except:
        pass

    # try repair
    try:
        fixed = repair_json(json_str)
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(f"Still invalid JSON after repair:\n{json_str}\nError: {str(e)}")


# Main tool
def parse_features(input_data):

    text = input_data.get("text", "")

    if not text:
        return {
            "status": "error",
            "message": "No input text"
        }

    try:
        raw = parse(text)

        if isinstance(raw, dict):
            features = raw

        elif isinstance(raw, str):
            features = extract_json(raw)

        else:
            raise ValueError("Unexpected parser output")

        # Clean values
        cleaned = {}

        for k, v in features.items():
            try:
                cleaned[k] = float(v)
            except:
                continue

        if not cleaned:
            return {
                "status": "error",
                "message": "No valid numeric features extracted"
            }

        return {
            "status": "ok",
            "features": cleaned
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Parse failed: {str(e)}"
        }