from tools.fetch_patient_tool import get_patient_features
from tools.predict_tool import predict_multimodal

patient = get_patient_features("R01-100")

print(patient)

features = patient["features"]

result = predict_multimodal(features)

print(result)