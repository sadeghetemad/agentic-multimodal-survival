from services.feature_completion_service import complete


def complete_features(input_data):

    features = input_data.get("features", {})

    if not features:
        return {
            "status": "error",
            "message": "No features provided"
        }

    try:
        completed = complete(features)
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    return {
        "status": "ok",
        "features": completed
    }