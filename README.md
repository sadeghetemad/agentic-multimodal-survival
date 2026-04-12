# NSCLC Multimodal AI Agent Prediction System

## Overview

This project is an **AI-powered clinical decision support system** for
**Non-Small Cell Lung Cancer (NSCLC)** survival prediction using
**multimodal data**:

-   Clinical data (e.g. age, smoking history)
-   Genomic data (e.g. mutations, biomarkers)
-   Imaging-derived features

It integrates: - AWS SageMaker (model serving) - AWS Feature Store
(data) - Athena (data retrieval) - Bedrock LLM (reasoning +
explanation) - LangGraph (agent orchestration)

------------------------------------------------------------------------

## Architecture

### Agent Pipeline

1.  **User Input**
2.  **Planner (LLM)**
3.  **Tool Execution**
4.  **Prediction**
5.  **Explanation (LLM)**

Core orchestration is handled via a graph-based agent:

-   Planner → decides steps
-   Tools → fetch/parse/validate/complete/predict
-   Response → formatted output

------------------------------------------------------------------------

## Project Structure

    agent/
      agent.py              # Main agent interface
      graph.py              # LangGraph workflow
      planner.py            # LLM-based planning
      executor.py           # Tool execution engine
      llm.py                # Bedrock integration

    tools/
      predict_tool.py
      fetch_patient_tool.py
      parse_features_tool.py
      validate_features_tool.py
      complete_features_tool.py

    services/
      feature_service.py
      feature_parser_service.py
      feature_validator_service.py
      feature_completion_service.py
      prediction_service.py

    app/
      streamlit_app.py      # UI

    config/
      settings.py

------------------------------------------------------------------------

## Data Sources

Multimodal data stored in AWS:

-   **Genomic Feature Group**
-   **Clinical Feature Group**
-   **Imaging Feature Group**

Joined via Athena queries.

------------------------------------------------------------------------

## Features

### 1. Intelligent Agent

-   Automatically understands user intent
-   Plans execution steps dynamically

### 2. Multimodal Prediction

-   Combines clinical + genomic + imaging
-   Uses PCA + XGBoost model

### 3. Feature Completion

-   Missing data filled using nearest neighbor similarity

### 4. Explainability

-   Feature importance (model-based)
-   Clinical reasoning (LLM-generated)

------------------------------------------------------------------------

## Example Usage

### Input

    65 year old smoker with tumor size 4.5 cm

### Output

    Risk: high
    Probability: 0.78

    Top Features:
    - tumor_size
    - smoking
    - age

    AI Analysis:
    Patient is high risk due to large tumor size and smoking history.

------------------------------------------------------------------------

## How It Works

### Planning

LLM classifies input: - MEDICAL → run pipeline - OUT_OF_SCOPE → reject

### Execution Flow

    parse → validate → complete → predict

### Prediction Pipeline

1.  Feature normalization
2.  PCA transformation
3.  XGBoost inference (SageMaker endpoint)

------------------------------------------------------------------------

## Setup

### Requirements

-   Python 3.11+
-   AWS credentials configured
-   SageMaker endpoint deployed

### Install

``` bash
pip install -r requirements.txt
```

### Run App

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## AWS Configuration

Required: - S3 bucket (artifacts) - Feature Store (3 groups) - Athena
enabled - SageMaker endpoint

------------------------------------------------------------------------

## Notes

-   Designed for research / clinical decision support
-   Not a replacement for medical diagnosis
-   Extendable to other cancers

------------------------------------------------------------------------

## Future Work

-   Add survival time prediction
-   Integrate radiology images directly (CNN/ViT)
-   Improve feature engineering
-   Integrate with Amazon Bedrock AgentCore

