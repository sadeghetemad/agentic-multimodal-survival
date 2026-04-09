
def explain_prediction(features, prediction):
    explanation = f"""
    The patient is high risk mainly due to:
    - tumor size: {features.get("tumor_size")}
    - age: {features.get("age")}
    """
    return explanation