# NSCLC Multimodal AI Agent (AgentCore + Bedrock)

## Overview

This project is an **AI-powered clinical decision support system** for
**Non-Small Cell Lung Cancer (NSCLC)** survival prediction using **multimodal data**:

- Clinical data (age, smoking, staging)
- Genomic data (mutations, biomarkers)
- Imaging-derived features

It integrates modern AWS services:

- Amazon SageMaker (model inference)
- AWS Feature Store + Athena (data layer)
- Amazon Bedrock (LLM reasoning)
- Amazon Bedrock AgentCore Runtime (execution layer)
- LangGraph (agent orchestration)

---

## 🧠 System Capability

The system supports **two input modes**:

### 1. Patient-based input
If a patient ID is provided:

- Features are retrieved from the database (Feature Store via Athena)
- The exact model input is constructed automatically
- Prediction is executed directly

---

### 2. Feature-based input (free text)
If the user provides clinical text:

The system:
- extracts structured features using LLM
- validates them
- completes missing values
- constructs the exact model input
- runs prediction

---

## 🔄 Agent Flow

```
User → Router → Fetch / Parse → Validate → Complete → Predict → Explain → Response
```

---

## ⚙️ Architecture Layers

| Layer | Responsibility |
|------|--------------|
| AgentCore Runtime | Execution + orchestration |
| LangGraph | Decision flow |
| Tools | Task execution |
| Services | Business logic |
| Bedrock LLM | Reasoning |

---

## 📁 Project Structure

```
agent/
  graph.py              # LangGraph workflow (core logic)
  llm.py                # Bedrock LLM wrapper

app/
  streamlit_app.py         # Streamlit app (UI)
  agentcore_app.py         # AgentCore app entrypoint

services/
  feature_service.py
  feature_parser_service.py
  feature_validator_service.py
  feature_completion_service.py
  prediction_pipeline.py
  prediction_service.py

tools/
  langchain_tools.py    # Agent tools


artifacts/
  xgboost-model
  scaler.joblib
  pca.joblib

config/
  settings.py
```

---

## 🧩 Agent Logic (LangGraph)

### Routing Decision

The agent uses an LLM to classify input:

- `"patient"` → use database
- `"text"` → extract features

---

### Patient Flow

```
route → fetch → predict → respond
```

Steps:
- Retrieve patient features from Athena
- Build model input
- Run prediction

---

### Text Flow

```
route → parse → validate → complete → predict → respond
```

Steps:
1. Extract features from text
2. Validate schema and values
3. Complete missing features
4. Build final model input
5. Run prediction

---

## 🔬 Prediction Pipeline

```
Input → Features → Scaling → PCA → XGBoost → SHAP → LLM Explanation
```

---

## 📊 Explainability

### 1. Model-level (SHAP)

- Feature contribution scores
- Mapped back from PCA space to original features

---

### 2. LLM-level (Clinical reasoning)

- Explains why risk is high/low
- Uses top contributing features
- Provides cautious interpretation

---

## 🤖 AgentCore Runtime

### Entry Point

```
aws_agentcore/app.py
```

Handles:
- Request lifecycle
- State passing
- Tool execution

---

### Execution Flow

```
AgentCore → Graph → Tools → Services → Model → LLM → Response
```

---

## 📈 Example

### Input

```
R01-029
```

### Output

```
Risk: low
Probability: 0.47

Top Features:
- pleural invasion (no)
- tumor stage
- genomic markers

AI Analysis:
Patient shows relatively low mortality risk due to absence of pleural invasion...
```

---

## 🚀 Setup

### Requirements

- Python 3.10+
- AWS credentials configured
- SageMaker endpoint deployed
- Bedrock access enabled

---

### Install

```bash
pip install -r requirements.txt
```

---

### Run UI

```bash
streamlit run app/streamlit_app.py
```

---

## ☁️ AWS Requirements

- S3 bucket (artifacts)
- Feature Store (multiple groups)
- Athena query access
- SageMaker endpoint
- Bedrock model access
- AgentCore runtime configured

---

## ⚠️ Notes

- This is a research / decision support tool
- Not intended for diagnosis
- Outputs must be interpreted by professionals

---

## 🔮 Future Work

- Deep multimodal models (ViT + omics)
- Fully managed AgentCore workflows