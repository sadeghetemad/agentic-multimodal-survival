from tools.predict import predict_multimodal

def simulate_change(features, changes):
    new_features = features.copy()
    new_features.update(changes)

    new_pred = predict_multimodal(new_features)
    return new_features, new_pred