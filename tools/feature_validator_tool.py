from services.feature_validator_service import validate


def validate_features(input_data):

    features = input_data.get("features", {})

    if not features:
        return {
            "status": "error",
            "message": "Empty features"
        }

    errors = validate(features)

    if errors:
        return {
            "status": "error",
            "message": errors
        }

    return {
        "status": "ok",
        "features": features
    }