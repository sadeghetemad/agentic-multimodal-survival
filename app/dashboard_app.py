import os
import re
import sys
import json
import uuid
import io
import gzip
import html
from textwrap import dedent
from pathlib import Path
from urllib.parse import quote_plus
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
import boto3
from botocore.exceptions import ClientError


# =======================================================
# Project path
# =======================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =======================================================
# Lightweight imports only
# =======================================================
from config.settings import (
    AWS_REGION,
    GENOMIC_FG,
    CLINICAL_FG,
    IMAGING_FG,
    BUCKET,
    PREFIX,
)
from services.feature_service import PatientFeatureService


# =======================================================
# Constants
# =======================================================
OHIF_BASE_URL = os.getenv(
    "OHIF_BASE_URL",
    "https://dxhhnitg56xjv.cloudfront.net/",
)

AGENTCORE_URL = os.getenv(
    "AGENTCORE_URL",
    "http://localhost:8080/invocations",
)

DEFAULT_PATIENT_ID = os.getenv("DEFAULT_PATIENT_ID", "R01-029")
DEFAULT_ACTOR_ID = os.getenv("DEFAULT_ACTOR_ID", "clinician_dashboard_1")

CPI_LOGO_LIGHT_PATH = Path(os.getenv("CPI_LOGO_LIGHT_PATH", str(CURRENT_DIR / "cpi_logo_blue.png")))
CPI_LOGO_DARK_PATH = Path(os.getenv("CPI_LOGO_DARK_PATH", str(CURRENT_DIR / "cpi_logo_dark.png")))

THUMBNAIL_BUCKET = os.getenv(
    "THUMBNAIL_BUCKET",
    "nsclc-medical-image-data-811165582441-eu-west-2-an",
)

THUMBNAIL_PREFIX = os.getenv(
    "THUMBNAIL_PREFIX",
    "processed/nsclc_radiogenomics/PNG",
)

# Optional imaging metadata source.
# Expected formats are JSON/CSV files containing DICOM-like fields such as:
# StudyDate, SeriesDate, AcquisitionDate, ContentDate, StudyInstanceUID,
# SeriesInstanceUID, SeriesDescription, Modality.
IMAGING_METADATA_BUCKET = os.getenv(
    "IMAGING_METADATA_BUCKET",
    THUMBNAIL_BUCKET,
)

IMAGING_METADATA_PREFIX = os.getenv(
    "IMAGING_METADATA_PREFIX",
    "processed/nsclc_radiogenomics/metadata",
)

# AWS HealthImaging datastore used by OHIF.
# If the datastore ID is not known, the app can resolve it from the datastore name.
HEALTHIMAGING_DATASTORE_ID = os.getenv("HEALTHIMAGING_DATASTORE_ID", "")
HEALTHIMAGING_DATASTORE_NAME = os.getenv(
    "HEALTHIMAGING_DATASTORE_NAME",
    "ahi-ohif-oidc-202605051517",
)


# =======================================================
# Feature metadata based on your uploaded 215 feature list
# =======================================================
GENE_FEATURES = {
    "lrig1", "hpgd", "gdf15", "cdh2", "postn", "vcan", "pdgfra",
    "vcam1", "cd44", "cd48", "cd4", "lyl1", "spi1", "cd37", "vim",
    "lmo2", "egr2", "bgn", "col4a1", "col5a1", "col5a2",
}

FEATURE_NAME_OVERRIDES = {
    "patientaffiliation_stanford": "Patient Affiliation: Stanford",
    "patientaffiliation_va": "Patient Affiliation: VA",

    "gender_female": "Gender: Female",
    "gender_male": "Gender: Male",
    "ethnicity_africanamerican": "Ethnicity: African American",
    "ethnicity_asian": "Ethnicity: Asian",
    "ethnicity_caucasian": "Ethnicity: Caucasian",
    "ethnicity_hispaniclatino": "Ethnicity: Hispanic / Latino",
    "ethnicity_nativehawaiianpacificislander": "Ethnicity: Native Hawaiian / Pacific Islander",

    "smokingstatus_current": "Smoking Status: Current Smoker",
    "smokingstatus_former": "Smoking Status: Former Smoker",
    "smokingstatus_nonsmoker": "Smoking Status: Non-smoker",

    "gg_0": "Ground Glass Opacity: 0%",
    "gg_025": "Ground Glass Opacity: 0–25%",
    "gg_100": "Ground Glass Opacity: 100%",
    "gg_2550": "Ground Glass Opacity: 25–50%",
    "gg_5075": "Ground Glass Opacity: 50–75%",
    "gg_75100": "Ground Glass Opacity: 75–100%",
    "gg_notassessed": "Ground Glass Opacity: Not Assessed",

    "tumorlocationchoicerul_checked": "Tumour Location: Right Upper Lobe",
    "tumorlocationchoicerul_unchecked": "Tumour Location Not Selected: Right Upper Lobe",
    "tumorlocationchoicerml_checked": "Tumour Location: Right Middle Lobe",
    "tumorlocationchoicerml_unchecked": "Tumour Location Not Selected: Right Middle Lobe",
    "tumorlocationchoicerll_checked": "Tumour Location: Right Lower Lobe",
    "tumorlocationchoicerll_unchecked": "Tumour Location Not Selected: Right Lower Lobe",
    "tumorlocationchoicelul_checked": "Tumour Location: Left Upper Lobe",
    "tumorlocationchoicelul_unchecked": "Tumour Location Not Selected: Left Upper Lobe",
    "tumorlocationchoicelll_checked": "Tumour Location: Left Lower Lobe",
    "tumorlocationchoicelll_unchecked": "Tumour Location Not Selected: Left Lower Lobe",
    "tumorlocationchoicellingula_checked": "Tumour Location: Lingula",
    "tumorlocationchoicellingula_unchecked": "Tumour Location Not Selected: Lingula",
    "tumorlocationchoiceunknown_unchecked": "Tumour Location: Unknown Not Selected",

    "histology_adenocarcinoma": "Histology: Adenocarcinoma",
    "histology_nsclcnosnototherwisespecified": "Histology: NSCLC NOS",
    "histology_squamouscellcarcinoma": "Histology: Squamous Cell Carcinoma",

    "pathologicaltstage_t1a": "Pathological T Stage: T1a",
    "pathologicaltstage_t1b": "Pathological T Stage: T1b",
    "pathologicaltstage_t2a": "Pathological T Stage: T2a",
    "pathologicaltstage_t2b": "Pathological T Stage: T2b",
    "pathologicaltstage_t3": "Pathological T Stage: T3",
    "pathologicaltstage_t4": "Pathological T Stage: T4",
    "pathologicaltstage_tis": "Pathological T Stage: Tis",
    "pathologicalnstage_n0": "Pathological N Stage: N0",
    "pathologicalnstage_n1": "Pathological N Stage: N1",
    "pathologicalnstage_n2": "Pathological N Stage: N2",
    "pathologicalmstage_m0": "Pathological M Stage: M0",
    "pathologicalmstage_m1a": "Pathological M Stage: M1a",
    "pathologicalmstage_m1b": "Pathological M Stage: M1b",

    "histopathologicalgrade_g1welldifferentiated": "Histopathological Grade: G1 Well Differentiated",
    "histopathologicalgrade_g2moderatelydifferentiated": "Histopathological Grade: G2 Moderately Differentiated",
    "histopathologicalgrade_g3poorlydifferentiated": "Histopathological Grade: G3 Poorly Differentiated",
    "histopathologicalgrade_othertypeiwelltomoderatelydifferentiated": "Histopathological Grade: Other Type I, Well to Moderately Differentiated",
    "histopathologicalgrade_othertypeiimoderatelytopoorlydifferen": "Histopathological Grade: Other Type II, Moderately to Poorly Differentiated",

    "lymphovascularinvasion_absent": "Lymphovascular Invasion: Absent",
    "lymphovascularinvasion_notcollected": "Lymphovascular Invasion: Not Collected",
    "lymphovascularinvasion_present": "Lymphovascular Invasion: Present",
    "pleuralinvasionelasticvisceralorparietal_no": "Pleural Invasion: No",
    "pleuralinvasionelasticvisceralorparietal_notcollected": "Pleural Invasion: Not Collected",
    "pleuralinvasionelasticvisceralorparietal_yes": "Pleural Invasion: Yes",

    "egfrmutationstatus_mutant": "EGFR Mutation Status: Mutant",
    "egfrmutationstatus_notcollected": "EGFR Mutation Status: Not Collected",
    "egfrmutationstatus_unknown": "EGFR Mutation Status: Unknown",
    "egfrmutationstatus_wildtype": "EGFR Mutation Status: Wild Type",
    "krasmutationstatus_mutant": "KRAS Mutation Status: Mutant",
    "krasmutationstatus_notcollected": "KRAS Mutation Status: Not Collected",
    "krasmutationstatus_unknown": "KRAS Mutation Status: Unknown",
    "krasmutationstatus_wildtype": "KRAS Mutation Status: Wild Type",
    "alktranslocationstatus_notcollected": "ALK Translocation Status: Not Collected",
    "alktranslocationstatus_translocated": "ALK Translocation Status: Translocated",
    "alktranslocationstatus_unknown": "ALK Translocation Status: Unknown",
    "alktranslocationstatus_wildtype": "ALK Translocation Status: Wild Type",

    "adjuvanttreatment_no": "Adjuvant Treatment: No",
    "adjuvanttreatment_yes": "Adjuvant Treatment: Yes",
    "chemotherapy_no": "Chemotherapy: No",
    "chemotherapy_yes": "Chemotherapy: Yes",
    "radiation_no": "Radiation Therapy: No",
    "radiation_yes": "Radiation Therapy: Yes",
    "recurrence_no": "Recurrence: No",
    "recurrence_yes": "Recurrence: Yes",
    "recurrencelocation_distant": "Recurrence Location: Distant",
    "recurrencelocation_local": "Recurrence Location: Local",
    "recurrencelocation_regional": "Recurrence Location: Regional",

    "ageathistologicaldiagnosis": "Age at Histological Diagnosis",
    "weightlbs": "Weight (lbs)",
    "packyears": "Pack Years",
    "timetodeathdays": "Time to Death (days)",
    "daysbetweenctandsurgery": "Days Between CT and Surgery",
    "survivalstatus": "Survival Status",
    "survival_status": "Survival Status",
}

RADIOMICS_FAMILIES = {
    "shape": "Shape",
    "firstorder": "First-order Intensity",
    "glcm": "GLCM Texture",
    "gldm": "GLDM Texture",
    "glrlm": "GLRLM Texture",
    "glszm": "GLSZM Texture",
    "ngtdm": "NGTDM Texture",
}

ACRONYMS = [
    "NSCLC", "NOS", "EGFR", "KRAS", "ALK", "GLCM", "GLDM",
    "GLRLM", "GLSZM", "NGTDM", "CT", "VA", "CD4", "CD44", "CD48",
]


# =======================================================
# Streamlit config
# =======================================================
st.set_page_config(
    page_title="NSCLC Clinical Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =======================================================
# CSS / UI polish
# =======================================================
def inject_css(theme: str) -> None:
    dark_tokens = """
        --bg:#07111f;
        --surface:#0d1b2e;
        --surface2:#10243d;
        --card:#0f2238;
        --text:#f8fafc;
        --muted:#9aa8bd;
        --border:#223955;
        --primary:#60a5fa;
        --primary2:#3b82f6;
        --primary-soft:rgba(96,165,250,.16);
        --button-text:#ffffff;
        --green:#22c55e;
        --red:#ef4444;
        --amber:#f59e0b;
        --input-bg:#0a1728;
        --tab-bg:#0b1a2c;
        --shadow:0 18px 50px rgba(0,0,0,.28);
    """

    light_tokens = """
        --bg:#f3f7fb;
        --surface:#ffffff;
        --surface2:#f8fafc;
        --card:#ffffff;
        --text:#172033;
        --muted:#5f6f89;
        --border:#d9e2ef;
        --primary:#1e5bff;
        --primary2:#1747c9;
        --primary-soft:#eaf1ff;
        --button-text:#ffffff;
        --green:#137a3d;
        --red:#b42318;
        --amber:#b45309;
        --input-bg:#ffffff;
        --tab-bg:#ffffff;
        --shadow:0 14px 36px rgba(31,41,55,.08);
    """

    if theme == "Dark":
        tokens = dark_tokens
        system_dark_css = ""
    elif theme == "Light":
        tokens = light_tokens
        system_dark_css = ""
    else:
        # System mode: light by default, dark when the OS/browser prefers dark.
        tokens = light_tokens
        system_dark_css = f"""
        @media (prefers-color-scheme: dark) {{
            :root {{
                {dark_tokens}
            }}
        }}
        """

    st.markdown(
        f"""
        <style>
        :root {{ {tokens} }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}

        .block-container {{
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }}

        h1,h2,h3,h4,h5,p,span,label,div {{
            color: var(--text);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            width: 285px !important;
            min-width: 285px !important;
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] > div {{
            background: var(--surface);
            padding: 1rem .9rem 1.2rem .9rem;
        }}

        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stCheckbox label,
        section[data-testid="stSidebar"] .stTextInput label {{
            font-size: 12px !important;
            color: var(--muted) !important;
            font-weight: 750 !important;
        }}

        .sidebar-brand {{
            background: linear-gradient(135deg, var(--primary-soft), var(--surface2));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 12px;
            margin-bottom: 14px;
        }}

        .sidebar-brand-row {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .sidebar-logo {{
            width: 38px;
            height: 38px;
            border-radius: 14px;
            background: var(--primary);
            color: white !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex: 0 0 auto;
        }}

        .sidebar-title {{
            font-size: 15px;
            font-weight: 950;
            line-height: 1.15;
            letter-spacing: -.02em;
        }}

        .sidebar-subtitle {{
            color: var(--muted);
            font-size: 11px;
            margin-top: 2px;
            line-height: 1.35;
        }}

        .sidebar-section {{
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }}

        .sidebar-section-title {{
            color: var(--muted);
            font-size: 11px;
            font-weight: 900;
            letter-spacing: .06em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .sidebar-help {{
            color: var(--muted);
            font-size: 11.5px;
            line-height: 1.45;
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: 13px;
            padding: 9px 10px;
            margin-top: 8px;
        }}

        .sidebar-status {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 9px 10px;
            border-radius: 13px;
            border: 1px solid var(--border);
            background: var(--surface2);
            font-size: 12px;
            font-weight: 800;
            margin-top: 8px;
        }}

        .status-dot {{
            width: 9px;
            height: 9px;
            border-radius: 999px;
            background: var(--green);
            display: inline-block;
        }}

        /* Inputs */
        .stTextInput > div > div > input {{
            background: var(--input-bg) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            height: 38px !important;
            font-size: 13px !important;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(30,91,255,.12) !important;
        }}

        .stSelectbox > div > div,
        .stMultiSelect > div > div {{
            background: var(--input-bg) !important;
            color: var(--text) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 12px !important;
            font-weight: 900 !important;
            border: 1px solid var(--primary2) !important;
            background: var(--primary) !important;
            color: var(--button-text) !important;
            box-shadow: 0 8px 18px rgba(30,91,255,.22) !important;
            min-height: 38px !important;
        }}

        .stButton > button:hover {{
            background: var(--primary2) !important;
            color: white !important;
            transform: translateY(-1px);
            border-color: var(--primary2) !important;
        }}

        .stDownloadButton > button {{
            border-radius: 12px !important;
            font-weight: 850 !important;
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            min-height: 38px !important;
        }}

        .stDownloadButton > button:hover {{
            border-color: var(--primary) !important;
            color: var(--primary) !important;
        }}

        .stLinkButton > a {{
            border-radius: 12px !important;
            background: var(--primary) !important;
            color: white !important;
            border: 1px solid var(--primary2) !important;
            font-weight: 900 !important;
        }}

        /* Hero */
        .hero {{
            background: linear-gradient(135deg, var(--surface), var(--surface2));
            border: 1px solid var(--border);
            border-radius: 26px;
            padding: 24px 28px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }}

        .hero-title {{
            font-size: 34px;
            font-weight: 950;
            letter-spacing: -.045em;
            margin-bottom: 6px;
        }}

        .hero-subtitle {{
            color: var(--muted);
            font-size: 14px;
            line-height: 1.55;
            max-width: 920px;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 11px;
            background: var(--primary-soft);
            border: 1px solid var(--border);
            border-radius: 999px;
            font-size: 12px;
            font-weight: 850;
            color: var(--primary);
            margin-right: 6px;
            margin-top: 12px;
        }}

        /* Cards */
        .metric-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(16,24,40,.055);
            height: 138px;
            min-height: 138px;
            max-height: 138px;
            overflow: hidden;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 11px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: .05em;
            line-height: 1.25;
            margin-bottom: 4px;
        }}

        .metric-value {{
            font-size: 24px;
            font-weight: 950;
            letter-spacing: -.035em;
            margin-top: 4px;
            line-height: 1.08;
            overflow-wrap: anywhere;
            word-break: break-word;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}

        .metric-help {{
            color: var(--muted);
            font-size: 11.5px;
            margin-top: 7px;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}

        .panel {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(16,24,40,.05);
            margin-bottom: 16px;
        }}

        .panel-title {{
            font-size: 18px;
            font-weight: 950;
            letter-spacing: -.02em;
            margin-bottom: 4px;
        }}

        .panel-subtitle {{
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 14px;
            line-height: 1.45;
        }}

        .small-muted {{
            color: var(--muted);
            font-size: 12px;
            line-height: 1.45;
        }}



        .metric-card-icon-layout {{
            display: grid;
            grid-template-columns: 44px minmax(0, 1fr);
            gap: 14px;
            align-items: center;
            height: 100%;
        }}

        .metric-card-icon-layout > div:last-child {{
            min-width: 0;
        }}

        .metric-icon {{
            width: 38px;
            height: 38px;
            border-radius: 13px;
            background: linear-gradient(135deg, var(--primary-soft), var(--surface2));
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            line-height: 1;
            box-shadow: 0 8px 20px rgba(30,91,255,.08);
            flex: 0 0 auto;
        }}

        .compact-sidebar-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px;
            box-shadow: 0 10px 28px rgba(16,24,40,.05);
            margin-bottom: 12px;
        }}

        .compact-sidebar-title {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 15px;
            font-weight: 950;
            margin-bottom: 8px;
        }}

        .compact-sidebar-text {{
            color: var(--muted);
            font-size: 12px;
            line-height: 1.5;
            margin-bottom: 10px;
        }}

        .theme-row-note {{
            color: var(--muted);
            font-size: 11px;
            margin: 2px 0 8px 0;
        }}

        .image-action-panel {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 16px;
            box-shadow: 0 8px 24px rgba(16,24,40,.05);
            margin-top: 0;
        }}

        .image-action-title {{
            font-size: 17px;
            font-weight: 950;
            margin-bottom: 6px;
        }}

        .image-action-text {{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
            margin-bottom: 10px;
        }}

        .image-thumb-wrap {{
            max-width: 440px;
        }}

        .image-action-panel {{
            max-width: 520px;
        }}


        .ai-risk-card {{
            background: linear-gradient(135deg, var(--card), var(--surface2));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 20px;
            box-shadow: 0 12px 34px rgba(16,24,40,.07);
            min-height: 150px;
        }}

        .ai-risk-top {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}

        .ai-risk-icon {{
            width: 46px;
            height: 46px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--primary-soft);
            border: 1px solid var(--border);
            font-size: 24px;
        }}

        .ai-risk-label {{
            color: var(--muted);
            font-size: 12px;
            font-weight: 900;
            letter-spacing: .06em;
            text-transform: uppercase;
        }}

        .ai-risk-value {{
            font-size: 34px;
            font-weight: 950;
            line-height: 1;
            letter-spacing: -.04em;
        }}

        .ai-risk-help {{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
            margin-top: 8px;
        }}

        .ai-report-note {{
            background: var(--primary-soft);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px 16px;
            color: var(--text);
            font-size: 14px;
            line-height: 1.5;
            margin-top: 16px;
        }}


        .clinical-timeline-wrap {{
            position: relative;
            margin: 18px 0 10px 0;
            padding-left: 18px;
        }}

        .clinical-timeline-line {{
            position: absolute;
            left: 31px;
            top: 8px;
            bottom: 8px;
            width: 2px;
            background: linear-gradient(180deg, var(--primary), var(--border));
            opacity: .35;
        }}

        .clinical-timeline-item {{
            position: relative;
            display: grid;
            grid-template-columns: 76px 1fr;
            gap: 16px;
            margin-bottom: 14px;
            align-items: stretch;
        }}

        .clinical-timeline-day {{
            font-size: 12px;
            font-weight: 900;
            color: var(--muted);
            padding-top: 14px;
            text-align: right;
        }}

        .clinical-timeline-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 13px 15px;
            box-shadow: 0 8px 22px rgba(16,24,40,.05);
        }}

        .clinical-timeline-dot {{
            position: absolute;
            left: 6px;
            top: 17px;
            width: 26px;
            height: 26px;
            border-radius: 999px;
            background: var(--primary-soft);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }}

        .clinical-timeline-title {{
            font-size: 15px;
            font-weight: 950;
            margin-bottom: 4px;
        }}

        .clinical-timeline-meta {{
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 6px;
        }}

        .clinical-timeline-detail {{
            color: var(--text);
            font-size: 13px;
            line-height: 1.45;
        }}

        .clinical-timeline-summary {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 16px 0;
        }}

        .clinical-timeline-mini {{
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px 12px;
        }}

        .clinical-timeline-mini-label {{
            color: var(--muted);
            font-size: 11px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: .05em;
        }}

        .clinical-timeline-mini-value {{
            font-size: 18px;
            font-weight: 950;
            margin-top: 3px;
        }}


        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: transparent;
            border-bottom: 1px solid var(--border);
        }}

        .stTabs [data-baseweb="tab"] {{
            background: var(--tab-bg);
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: 12px 12px 0 0;
            padding: 8px 12px;
            color: var(--muted);
            font-weight: 850;
            font-size: 13px;
        }}

        .stTabs [aria-selected="true"] {{
            color: var(--primary) !important;
            background: var(--primary-soft) !important;
        }}

        /* Cleaner multi-select chips and scrollable selected area */
        div[data-baseweb="select"] > div {{
            max-height: 132px !important;
            overflow-y: auto !important;
            align-content: flex-start !important;
        }}

        div[data-baseweb="select"] span {{
            font-size: 12px !important;
        }}

        div[data-baseweb="tag"] {{
            background: var(--primary-soft) !important;
            border: 1px solid var(--border) !important;
            color: var(--primary) !important;
            border-radius: 999px !important;
            max-width: 210px !important;
        }}

        div[data-baseweb="tag"] span {{
            color: var(--primary) !important;
            font-weight: 800 !important;
        }}

        div[data-baseweb="tag"] svg {{
            color: var(--primary) !important;
            fill: var(--primary) !important;
        }}

        /* DataFrames */
        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}

        /* Progress */
        div[data-testid="stProgress"] > div > div > div {{
            background-color: var(--primary);
        }}

        hr {{
            border-color: var(--border);
        }}

        a {{
            color: var(--primary);
        }}

        {system_dark_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =======================================================
# Lightweight services
# =======================================================
@st.cache_resource(show_spinner=False)
def get_feature_service() -> PatientFeatureService:
    return PatientFeatureService(
        region=AWS_REGION,
        genomic_fg_name=GENOMIC_FG,
        clinical_fg_name=CLINICAL_FG,
        imaging_fg_name=IMAGING_FG,
        bucket=BUCKET,
        prefix=PREFIX,
        use_online_store=True,
        cache_ttl_seconds=3600,
        cache_max_size=5000,
    )


@st.cache_data(show_spinner=False, ttl=600)
def fetch_patient_features(patient_id: str) -> Dict[str, Any]:
    service = get_feature_service()
    result = service.get_patient_features(patient_id)
    return result


# =======================================================
# Utilities
# =======================================================
def build_ohif_patient_url(patient_id: str) -> str:
    return f"{OHIF_BASE_URL}?mrn={quote_plus(patient_id)}"


@st.cache_data(show_spinner=False, ttl=600)
def fetch_thumbnail_from_s3(patient_id: str) -> Optional[bytes]:
    """
    Fetch CT thumbnail PNG from S3.

    Expected object:
    s3://nsclc-medical-image-data-811165582441-eu-west-2-an/
        processed/nsclc_radiogenomics/PNG/{patient_id}.png
    """
    if not patient_id:
        return None

    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"{THUMBNAIL_PREFIX.rstrip('/')}/{patient_id}.png"

    try:
        obj = s3.get_object(
            Bucket=THUMBNAIL_BUCKET,
            Key=key,
        )
        return obj["Body"].read()

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")

        if error_code in {"NoSuchKey", "404", "NotFound"}:
            return None

        raise



def get_cpi_logo_path(theme: str) -> Optional[Path]:
    """
    Return a CPI logo asset that fits the selected theme.
    Light/System uses the original white logo. Dark uses the transparent logo.
    """
    if theme == "Dark" and CPI_LOGO_DARK_PATH.exists():
        return CPI_LOGO_DARK_PATH
    if CPI_LOGO_LIGHT_PATH.exists():
        return CPI_LOGO_LIGHT_PATH
    if CPI_LOGO_DARK_PATH.exists():
        return CPI_LOGO_DARK_PATH
    return None


def render_streamlit_image(image_data: bytes, caption: str = "") -> None:
    """
    Streamlit renamed use_column_width to use_container_width in newer versions.
    This keeps the dashboard compatible with both, because apparently naming one
    argument consistently was too luxurious.
    """
    try:
        st.image(
            image_data,
            caption=caption,
            use_container_width=True,
        )
    except TypeError:
        st.image(
            image_data,
            caption=caption,
            use_column_width=True,
        )


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if str(value).strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def to_float_or_none(value: Any) -> Optional[float]:
    try:
        if is_missing(value):
            return None
        return float(value)
    except Exception:
        return None


def title_with_acronyms(text: str) -> str:
    out = text.title()
    for ac in ACRONYMS:
        out = re.sub(rf"\b{re.escape(ac.title())}\b", ac, out)
    return out


def split_camel_or_compact(name: str) -> str:
    compact_map = {
        "ageathistologicaldiagnosis": "Age at Histological Diagnosis",
        "timetodeathdays": "Time to Death (days)",
        "daysbetweenctandsurgery": "Days Between CT and Surgery",
        "weightlbs": "Weight (lbs)",
        "packyears": "Pack Years",
    }
    if name in compact_map:
        return compact_map[name]
    return title_with_acronyms(name.replace("_", " "))


def prettify_radiomics_name(feature_name: str) -> Tuple[str, str, str]:
    parts = feature_name.split("_")
    if len(parts) < 3:
        return "Imaging / Radiomics", "Radiomics", title_with_acronyms(feature_name.replace("_", " "))

    family_key = parts[1]
    metric = "_".join(parts[2:])
    family = RADIOMICS_FAMILIES.get(family_key, title_with_acronyms(family_key))

    metric_clean = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", metric)
    metric_clean = metric_clean.replace("2d", "2D").replace("3d", "3D")
    metric_clean = title_with_acronyms(metric_clean.replace("_", " "))

    if metric_clean.startswith("10"):
        metric_clean = "10th Percentile"
    elif metric_clean.startswith("90"):
        metric_clean = "90th Percentile"

    return "Imaging / Radiomics", family, f"{family}: {metric_clean}"


def feature_metadata(feature_name: str) -> Dict[str, str]:
    raw = str(feature_name).strip()
    low = raw.lower()

    if low in GENE_FEATURES:
        return {
            "display_name": low.upper(),
            "group": "Gene Expression",
            "subgroup": "Gene Signature",
        }

    if low in FEATURE_NAME_OVERRIDES:
        group = infer_feature_group(low)
        return {
            "display_name": FEATURE_NAME_OVERRIDES[low],
            "group": group,
            "subgroup": infer_feature_subgroup(low),
        }

    if low.startswith("original_"):
        group, subgroup, display = prettify_radiomics_name(low)
        return {
            "display_name": display,
            "group": group,
            "subgroup": subgroup,
        }

    return {
        "display_name": split_camel_or_compact(low),
        "group": infer_feature_group(low),
        "subgroup": infer_feature_subgroup(low),
    }


def infer_feature_group(name: str) -> str:
    n = name.lower()

    if n in GENE_FEATURES:
        return "Gene Expression"

    if n.startswith("original_"):
        return "Imaging / Radiomics"

    if n.startswith("patientaffiliation"):
        return "Dataset / Cohort"

    if n.startswith("gender") or n.startswith("ethnicity") or n in {"ageathistologicaldiagnosis", "weightlbs"}:
        return "Clinical / Demographics"

    if n.startswith("smokingstatus") or n == "packyears":
        return "Clinical / Smoking"

    if n.startswith("pathological") or n.startswith("histopathologicalgrade"):
        return "Clinical / Pathology & Stage"

    if n.startswith("histology"):
        return "Clinical / Histology"

    if n.startswith("tumorlocation") or n.startswith("gg_"):
        return "Clinical / Tumour Imaging Summary"

    if n.startswith("lymphovascular") or n.startswith("pleural"):
        return "Clinical / Invasion"

    if n.startswith("egfr") or n.startswith("kras") or n.startswith("alk"):
        return "Genomic / Biomarkers"

    if n.startswith("adjuvant") or n.startswith("chemotherapy") or n.startswith("radiation"):
        return "Clinical / Treatment"

    if n.startswith("recurrence") or n in {"timetodeathdays", "daysbetweenctandsurgery", "survivalstatus", "survival_status"}:
        return "Clinical / Outcome & Timeline"

    return "Unmapped"


def infer_feature_subgroup(name: str) -> str:
    n = name.lower()

    if n in GENE_FEATURES:
        return "Gene Signature"
    if n.startswith("original_shape"):
        return "Shape"
    if n.startswith("original_firstorder"):
        return "First-order Intensity"
    if n.startswith("original_glcm"):
        return "GLCM Texture"
    if n.startswith("original_gldm"):
        return "GLDM Texture"
    if n.startswith("original_glrlm"):
        return "GLRLM Texture"
    if n.startswith("original_glszm"):
        return "GLSZM Texture"
    if n.startswith("original_ngtdm"):
        return "NGTDM Texture"
    if n.startswith("pathological"):
        return "TNM Stage"
    if n.startswith("egfr") or n.startswith("kras") or n.startswith("alk"):
        return "Mutation / Translocation Status"
    if n.startswith("smoking") or n == "packyears":
        return "Smoking Exposure"
    if n.startswith("tumorlocation"):
        return "Tumour Location"
    if n in {"survivalstatus", "survival_status"}:
        return "Observed Outcome"
    if n.startswith("gg_"):
        return "Ground Glass Opacity"
    return "General"


def format_feature_value(value: Any) -> str:
    if is_missing(value):
        return ""
    numeric = to_float_or_none(value)
    if numeric is not None:
        if abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
        return f"{numeric:.4g}"
    return str(value)


def features_to_dataframe(features: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feature, value in features.items():
        meta = feature_metadata(feature)
        numeric_value = to_float_or_none(value)
        active = None
        if numeric_value is not None and numeric_value in {0.0, 1.0}:
            active = "Yes" if numeric_value == 1.0 else "No"

        rows.append(
            {
                "Display Name": meta["display_name"],
                "Raw Feature": str(feature),
                "Value": format_feature_value(value),
                "Numeric Value": numeric_value,
                "Active": active or "",
                "Group": meta["group"],
                "Subgroup": meta["subgroup"],
                "Missing": is_missing(value),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["Display Name", "Raw Feature", "Value", "Active", "Group", "Subgroup"]:
        df[col] = df[col].astype("string")

    df["Missing"] = df["Missing"].astype(bool)
    return df.sort_values(["Group", "Subgroup", "Display Name"]).reset_index(drop=True)


def active_one_hot_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["Active"] == "Yes") & (~df["Group"].isin(["Gene Expression", "Imaging / Radiomics"]))
    return df.loc[mask, ["Display Name", "Value", "Group", "Subgroup", "Raw Feature"]].copy()


def get_feature_row(df: pd.DataFrame, raw_names: list[str]) -> Optional[pd.Series]:
    raw_set = {x.lower() for x in raw_names}
    hit = df[df["Raw Feature"].str.lower().isin(raw_set)]
    if hit.empty:
        return None
    return hit.iloc[0]


def value_for(df: pd.DataFrame, raw_names: list[str], default: str = "N/A") -> str:
    row = get_feature_row(df, raw_names)
    if row is None:
        return default
    value = str(row.get("Value", "")).strip()
    return value if value else default


def active_value_for_prefix(df: pd.DataFrame, prefix: str, default: str = "N/A") -> str:
    subset = df[
        (df["Raw Feature"].str.lower().str.startswith(prefix.lower()))
        & (df["Active"] == "Yes")
    ]
    if subset.empty:
        return default
    label = str(subset.iloc[0]["Display Name"])
    if ":" in label:
        return label.split(":", 1)[1].strip()
    return label


def survival_status_label(df: pd.DataFrame) -> tuple[str, str]:
    value = value_for(df, ["survivalstatus", "survival_status"], default="N/A")
    if value == "N/A":
        return "N/A", "Not available"
    try:
        v = float(value)
        if v == 0:
            return "Alive", "Observed label: 0"
        if v == 1:
            return "Deceased", "Observed label: 1"
    except Exception:
        pass
    return value, "Observed outcome value"


def clinical_key_cards(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    survival_label, survival_help = survival_status_label(df)
    return [
        ("🧓 Age", value_for(df, ["ageathistologicaldiagnosis"]), "Age at histological diagnosis"),
        ("⚧ Gender", active_value_for_prefix(df, "gender_"), "Selected category"),
        ("🌍 Ethnicity", active_value_for_prefix(df, "ethnicity_"), "Selected category"),
        ("🚬 Smoking", active_value_for_prefix(df, "smokingstatus_"), "Smoking status"),
        ("📦 Pack Years", value_for(df, ["packyears"]), "Smoking exposure"),
        ("⚖️ Weight", value_for(df, ["weightlbs"]), "Weight in lbs"),
        ("🧬 Histology", active_value_for_prefix(df, "histology_"), "Tumour histology"),
        ("🎯 T Stage", active_value_for_prefix(df, "pathologicaltstage_"), "Pathological T stage"),
        ("🔗 N Stage", active_value_for_prefix(df, "pathologicalnstage_"), "Pathological N stage"),
        ("🧭 M Stage", active_value_for_prefix(df, "pathologicalmstage_"), "Pathological M stage"),
        ("❤️ Survival Status", survival_label, survival_help),
        ("⏱️ Time to Death", value_for(df, ["timetodeathdays"]), "Days, if available"),
    ]




def _normalise_dicom_date(value: Any) -> Optional[pd.Timestamp]:
    """
    Parse DICOM-style dates such as 20200131 or normal date strings.
    Returns None if the date is missing or unusable.
    """
    if is_missing(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    # DICOM DA format: YYYYMMDD
    if re.fullmatch(r"\d{8}", raw):
        try:
            return pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        except Exception:
            return None

    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed



def _safe_decompress_healthimaging_payload(payload: Any) -> bytes:
    """
    HealthImaging GetImageSetMetadata returns gzip-compressed JSON.
    Boto3 may expose it as a stream or as bytes depending on botocore internals.
    """
    if hasattr(payload, "read"):
        raw = payload.read()
    elif isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    else:
        raw = bytes(payload)

    try:
        return gzip.decompress(raw)
    except Exception:
        return raw


def _flatten_dict(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested HealthImaging/DICOM metadata, keeping leaf values.
    This is ugly, but so is real-world metadata. We cope.
    """
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            clean_key = str(key)
            next_prefix = f"{prefix}.{clean_key}" if prefix else clean_key
            out.update(_flatten_dict(value, next_prefix))
    elif isinstance(obj, list):
        # Avoid exploding giant per-frame arrays. Keep short lists only.
        if len(obj) <= 5:
            for i, value in enumerate(obj):
                out.update(_flatten_dict(value, f"{prefix}.{i}"))
    else:
        out[prefix] = obj

    return out


def _pick_first_value(flat: dict[str, Any], candidate_names: list[str]) -> Any:
    """
    Pick the first metadata value whose key ends with or equals one of candidate names.
    Works with normalized metadata and DICOM JSON-ish structures.
    """
    lowered = {k.lower(): v for k, v in flat.items()}

    for name in candidate_names:
        n = name.lower()
        if n in lowered and not is_missing(lowered[n]):
            return lowered[n]

    for key, value in lowered.items():
        for name in candidate_names:
            n = name.lower()
            if key.endswith("." + n) or key.endswith(n):
                if not is_missing(value):
                    return value

    return None


@st.cache_data(show_spinner=False, ttl=1200)
def resolve_healthimaging_datastore_id() -> str:
    """
    Resolve AWS HealthImaging datastore ID from either:
    1) HEALTHIMAGING_DATASTORE_ID env var
    2) HEALTHIMAGING_DATASTORE_NAME env var / default name
    """
    if HEALTHIMAGING_DATASTORE_ID:
        return HEALTHIMAGING_DATASTORE_ID

    client = boto3.client("medical-imaging", region_name=AWS_REGION)

    try:
        paginator = client.get_paginator("list_datastores")
        for page in paginator.paginate():
            for ds in page.get("datastoreSummaries", []):
                if ds.get("datastoreName") == HEALTHIMAGING_DATASTORE_NAME:
                    return ds.get("datastoreId", "")
    except Exception:
        return ""

    return ""


@st.cache_data(show_spinner=False, ttl=600)
def fetch_healthimaging_timeline(patient_id: str) -> pd.DataFrame:
    """
    Search AWS HealthImaging by DICOMPatientId and return imaging-study timeline rows.

    HealthImaging image sets usually map closely to DICOM Series. That means one patient
    can have multiple image sets across studies/series, and we can recover repeated imaging
    visits from StudyDate/SeriesDate/AcquisitionDate metadata. Finally, something useful
    from metadata, humanity briefly redeems itself.
    """
    if not patient_id:
        return pd.DataFrame()

    datastore_id = resolve_healthimaging_datastore_id()
    if not datastore_id:
        return pd.DataFrame()

    client = boto3.client("medical-imaging", region_name=AWS_REGION)

    search_filter = {
        "filters": [
            {
                "operator": "EQUAL",
                "values": [{"DICOMPatientId": patient_id}],
            }
        ]
    }

    summaries: list[dict[str, Any]] = []
    try:
        paginator = client.get_paginator("search_image_sets")
        for page in paginator.paginate(datastoreId=datastore_id, searchCriteria=search_filter):
            summaries.extend(page.get("imageSetsMetadataSummaries", []))
    except Exception:
        return pd.DataFrame()

    if not summaries:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for summary in summaries:
        image_set_id = summary.get("imageSetId", "")
        dicom_tags = summary.get("DICOMTags", {}) or {}

        # Summary-level tags are often enough.
        study_date = (
            dicom_tags.get("DICOMStudyDate")
            or dicom_tags.get("StudyDate")
            or dicom_tags.get("DICOMStudyDateAndTime", {}).get("DICOMStudyDate")
            if isinstance(dicom_tags.get("DICOMStudyDateAndTime"), dict)
            else None
        )

        series_date = dicom_tags.get("DICOMSeriesDate") or dicom_tags.get("SeriesDate")
        modality = dicom_tags.get("DICOMSeriesModality") or dicom_tags.get("Modality") or "CT"
        study_uid = dicom_tags.get("DICOMStudyInstanceUID") or ""
        series_uid = dicom_tags.get("DICOMSeriesInstanceUID") or ""
        study_desc = dicom_tags.get("DICOMStudyDescription") or ""
        body_part = dicom_tags.get("DICOMSeriesBodyPart") or ""
        series_number = dicom_tags.get("DICOMSeriesNumber") or ""

        metadata_date = _normalise_dicom_date(study_date or series_date)

        # If summary is missing date, fetch full normalized metadata.
        full_details = {}
        if metadata_date is None and image_set_id:
            try:
                meta_response = client.get_image_set_metadata(
                    datastoreId=datastore_id,
                    imageSetId=image_set_id,
                )
                payload = meta_response.get("imageSetMetadataBlob")
                if payload is not None:
                    decoded = _safe_decompress_healthimaging_payload(payload)
                    full_obj = json.loads(decoded.decode("utf-8"))
                    flat = _flatten_dict(full_obj)
                    full_details = flat

                    metadata_date = _normalise_dicom_date(
                        _pick_first_value(
                            flat,
                            [
                                "DICOMStudyDate",
                                "StudyDate",
                                "DICOMSeriesDate",
                                "SeriesDate",
                                "AcquisitionDate",
                                "ContentDate",
                            ],
                        )
                    )
                    modality = _pick_first_value(flat, ["Modality", "DICOMSeriesModality"]) or modality
                    study_uid = _pick_first_value(flat, ["StudyInstanceUID", "DICOMStudyInstanceUID"]) or study_uid
                    series_uid = _pick_first_value(flat, ["SeriesInstanceUID", "DICOMSeriesInstanceUID"]) or series_uid
                    study_desc = _pick_first_value(flat, ["StudyDescription", "DICOMStudyDescription"]) or study_desc
                    body_part = _pick_first_value(flat, ["BodyPartExamined", "DICOMSeriesBodyPart"]) or body_part
                    series_number = _pick_first_value(flat, ["SeriesNumber", "DICOMSeriesNumber"]) or series_number
            except Exception:
                pass

        if metadata_date is None:
            continue

        details = []
        if modality:
            details.append(str(modality))
        if study_desc and str(study_desc).lower() not in {"nan", "none"}:
            details.append(str(study_desc)[:50])
        if body_part and str(body_part).lower() not in {"nan", "none"}:
            details.append(f"Body part: {body_part}")
        if series_number and str(series_number).lower() not in {"nan", "none"}:
            details.append(f"Series: {series_number}")

        rows.append(
            {
                "Date": metadata_date.normalize(),
                "Event": "Imaging study",
                "Type": "Imaging",
                "Details": " | ".join(details) if details else "Imaging metadata from AWS HealthImaging",
                "ImageSetId": image_set_id,
                "StudyInstanceUID": str(study_uid),
                "SeriesInstanceUID": str(series_uid),
                "Source": "AWS HealthImaging",
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["Date", "ImageSetId", "StudyInstanceUID", "SeriesInstanceUID"])
    out = out.sort_values(["Date", "StudyInstanceUID", "SeriesInstanceUID"]).reset_index(drop=True)

    first_date = out["Date"].min()
    out["Day"] = (out["Date"] - first_date).dt.days.astype(int)

    # Make repeated imaging visually clear.
    out["Event"] = [
        f"Imaging visit {i + 1}" if len(out) > 1 else "Imaging visit"
        for i in range(len(out))
    ]

    return out[
        [
            "Day",
            "Date",
            "Event",
            "Type",
            "Details",
            "ImageSetId",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Source",
        ]
    ]


def _metadata_rows_from_json_bytes(payload: bytes) -> list[dict[str, Any]]:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception:
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for key in ["rows", "items", "studies", "series", "metadata"]:
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [obj]

    return []


def _metadata_rows_from_csv_bytes(payload: bytes) -> list[dict[str, Any]]:
    try:
        df = pd.read_csv(io.BytesIO(payload))
        return df.to_dict("records")
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=600)
def fetch_imaging_metadata_timeline(patient_id: str) -> pd.DataFrame:
    """
    Fetch optional imaging timeline metadata from S3.

    This function is intentionally tolerant:
    - JSON and CSV are both accepted.
    - Fields may be DICOM-like or slightly renamed.
    - If no metadata files exist, it returns an empty DataFrame quietly.

    In other words, unlike humans filling forms, it tries not to collapse immediately.
    """
    if not patient_id:
        return pd.DataFrame()

    s3 = boto3.client("s3", region_name=AWS_REGION)
    prefix = IMAGING_METADATA_PREFIX.rstrip("/")

    candidate_prefixes = [
        f"{prefix}/{patient_id}",
        f"{prefix}/{patient_id}/",
        prefix,
    ]

    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for candidate_prefix in candidate_prefixes:
        try:
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=IMAGING_METADATA_BUCKET,
                Prefix=candidate_prefix,
                PaginationConfig={"MaxItems": 300},
            )

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    if not key or key in seen_keys:
                        continue
                    if patient_id.lower() not in key.lower() and candidate_prefix == prefix:
                        continue
                    if not key.lower().endswith((".json", ".csv")):
                        continue

                    seen_keys.add(key)
                    try:
                        body = s3.get_object(Bucket=IMAGING_METADATA_BUCKET, Key=key)["Body"].read()
                    except Exception:
                        continue

                    if key.lower().endswith(".json"):
                        parsed_rows = _metadata_rows_from_json_bytes(body)
                    else:
                        parsed_rows = _metadata_rows_from_csv_bytes(body)

                    for r in parsed_rows:
                        r["_source_key"] = key
                        rows.append(r)

        except ClientError:
            continue
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    raw_df = pd.DataFrame(rows)

    # Keep rows for this patient if a patient column exists.
    patient_cols = [
        c for c in raw_df.columns
        if c.lower() in {"patientid", "patient_id", "mrn", "patient", "subjectid", "subject_id"}
    ]
    if patient_cols:
        col = patient_cols[0]
        raw_df = raw_df[raw_df[col].astype(str).str.lower().eq(patient_id.lower())].copy()

    if raw_df.empty:
        return pd.DataFrame()

    date_candidates = [
        "StudyDate", "study_date", "SeriesDate", "series_date",
        "AcquisitionDate", "acquisition_date", "ContentDate", "content_date",
        "InstanceCreationDate", "instance_creation_date",
    ]

    def row_date(row: pd.Series) -> Optional[pd.Timestamp]:
        for col in date_candidates:
            if col in row.index:
                parsed = _normalise_dicom_date(row.get(col))
                if parsed is not None and not pd.isna(parsed):
                    return parsed
        return None

    out_rows = []
    for _, row in raw_df.iterrows():
        date_value = row_date(row)
        if date_value is None or pd.isna(date_value):
            continue

        study_uid = str(row.get("StudyInstanceUID", row.get("study_uid", row.get("StudyUID", "")))).strip()
        series_uid = str(row.get("SeriesInstanceUID", row.get("series_uid", row.get("SeriesUID", "")))).strip()
        series_desc = str(row.get("SeriesDescription", row.get("series_description", row.get("Description", "")))).strip()
        modality = str(row.get("Modality", row.get("modality", "CT"))).strip() or "CT"

        label_bits = [modality]
        if series_desc and series_desc.lower() not in {"nan", "none"}:
            label_bits.append(series_desc[:45])

        out_rows.append(
            {
                "Date": date_value.normalize(),
                "Event": "Imaging study",
                "Type": "Imaging",
                "Details": " | ".join(label_bits),
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Source": row.get("_source_key", ""),
            }
        )

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)
    out = out.drop_duplicates(subset=["Date", "StudyInstanceUID", "SeriesInstanceUID", "Details"])
    out = out.sort_values("Date").reset_index(drop=True)

    first_date = out["Date"].min()
    out["Day"] = (out["Date"] - first_date).dt.days.astype(int)
    out["Event"] = [
        f"Imaging study {i + 1}" if len(out) > 1 else "Imaging study"
        for i in range(len(out))
    ]

    return out[["Day", "Date", "Event", "Type", "Details", "StudyInstanceUID", "SeriesInstanceUID", "Source"]]


def render_clinical_timeline(df: pd.DataFrame, patient_id: str) -> None:
    """
    Build a compact clinical timeline from available time-related fields.
    It is not a full longitudinal record, because the dataset is not exactly a Netflix series of patient events.
    """
    events = []
    imaging_meta_df = fetch_healthimaging_timeline(patient_id)

    # Fallback for older/local exports if HealthImaging does not return metadata.
    if imaging_meta_df.empty:
        imaging_meta_df = fetch_imaging_metadata_timeline(patient_id)

    ct_to_surgery = to_float_or_none(value_for(df, ["daysbetweenctandsurgery"], default=""))
    time_to_death = to_float_or_none(value_for(df, ["timetodeathdays"], default=""))

    recurrence = active_value_for_prefix(df, "recurrence_")
    recurrence_location = active_value_for_prefix(df, "recurrencelocation_")
    survival_label, _ = survival_status_label(df)

    # Baseline / diagnosis anchor
    events.append(
        {
            "Event": "Histological diagnosis",
            "Day": 0,
            "Type": "Diagnosis",
            "Details": "Clinical baseline and histological diagnosis",
        }
    )

    # CT/surgery interval if available
    if ct_to_surgery is not None:
        if ct_to_surgery >= 0:
            events.append(
                {
                    "Event": "Surgery",
                    "Day": ct_to_surgery,
                    "Type": "Treatment",
                    "Details": f"{ct_to_surgery:.0f} days after CT",
                }
            )
            events.append(
                {
                    "Event": "CT scan",
                    "Day": 0,
                    "Type": "Imaging",
                    "Details": "CT imaging reference point",
                }
            )
        else:
            events.append(
                {
                    "Event": "CT scan",
                    "Day": abs(ct_to_surgery),
                    "Type": "Imaging",
                    "Details": f"{abs(ct_to_surgery):.0f} days after surgery/diagnosis reference",
                }
            )

    # Treatment markers as known categorical events
    treatment_events = [
        ("Adjuvant treatment", active_value_for_prefix(df, "adjuvanttreatment_"), "Treatment", "💊"),
        ("Chemotherapy", active_value_for_prefix(df, "chemotherapy_"), "Treatment", "🧪"),
        ("Radiation therapy", active_value_for_prefix(df, "radiation_"), "Treatment", "☢️"),
    ]
    for name, value, typ, _icon in treatment_events:
        if value not in {"N/A", "No", "Not Collected", "Unknown"}:
            events.append(
                {
                    "Event": name,
                    "Day": max(1, int(ct_to_surgery or 1)),
                    "Type": typ,
                    "Details": f"Recorded as: {value}",
                }
            )

    # Recurrence marker, if recorded
    if recurrence not in {"N/A", "No", "Not Collected", "Unknown"}:
        detail = f"Recorded as: {recurrence}"
        if recurrence_location not in {"N/A", "Not Collected", "Unknown"}:
            detail += f" | Location: {recurrence_location}"
        events.append(
            {
                "Event": "Recurrence",
                "Day": max(30, int((time_to_death or 180) * 0.55)),
                "Type": "Outcome",
                "Details": detail,
            }
        )

    # Outcome marker
    if time_to_death is not None:
        events.append(
            {
                "Event": "Death" if survival_label == "Deceased" else "Last follow-up",
                "Day": time_to_death,
                "Type": "Outcome",
                "Details": f"{survival_label}; {time_to_death:.0f} days",
            }
        )
    else:
        events.append(
            {
                "Event": "Current observed status",
                "Day": max(60, int((ct_to_surgery or 0) + 60)),
                "Type": "Outcome",
                "Details": survival_label,
            }
        )

    if not imaging_meta_df.empty:
        for _, image_row in imaging_meta_df.iterrows():
            date_value = image_row.get("Date", "")
            date_text = ""
            try:
                date_text = pd.to_datetime(date_value).strftime("%Y-%m-%d")
            except Exception:
                date_text = str(date_value)

            events.append(
                {
                    "Event": image_row.get("Event", "Imaging study"),
                    "Day": int(image_row.get("Day", 0)),
                    "Type": "Imaging",
                    "Details": f"{date_text} | {image_row.get('Details', 'Imaging metadata')}",
                }
            )

    timeline_df = pd.DataFrame(events).drop_duplicates(subset=["Event", "Day", "Type", "Details"])
    timeline_df["Day"] = pd.to_numeric(timeline_df["Day"], errors="coerce").fillna(0)
    timeline_df = timeline_df.sort_values("Day").reset_index(drop=True)

    st.markdown("#### Clinical Timeline")
    if imaging_meta_df.empty:
        st.caption("A compact timeline reconstructed from available clinical timing fields and recorded imaging visits.")
    else:
        st.caption("A compact timeline reconstructed from clinical fields and recorded imaging visits.")

    # Clean clinician-facing timeline as cards.
    # Plotly was turning a simple timeline into interpretive dance, so cards it is.
    timeline_df = timeline_df.copy()
    timeline_df["Day"] = pd.to_numeric(timeline_df["Day"], errors="coerce").fillna(0).astype(int)
    timeline_df = timeline_df.sort_values(["Day", "Type", "Event"]).reset_index(drop=True)

    icon_map = {
        "Diagnosis": "🩺",
        "Imaging": "🖼️",
        "Treatment": "💊",
        "Outcome": "📌",
    }

    total_events = len(timeline_df)
    imaging_count = int((timeline_df["Type"] == "Imaging").sum())
    treatment_count = int((timeline_df["Type"] == "Treatment").sum())
    max_day = int(timeline_df["Day"].max()) if not timeline_df.empty else 0

    summary_html = f"""
        <div class="timeline-summary">
            <div class="mini"><div class="mini-label">Events</div><div class="mini-value">{total_events}</div></div>
            <div class="mini"><div class="mini-label">Imaging Visits</div><div class="mini-value">{imaging_count}</div></div>
            <div class="mini"><div class="mini-label">Treatment Events</div><div class="mini-value">{treatment_count}</div></div>
            <div class="mini"><div class="mini-label">Observed Days</div><div class="mini-value">{max_day}</div></div>
        </div>
    """

    items_html = []
    for _, row in timeline_df.iterrows():
        event_type = str(row.get("Type", "Event"))
        icon = icon_map.get(event_type, "•")
        day = int(row.get("Day", 0))
        event = html.escape(str(row.get("Event", "Event")))
        details = html.escape(str(row.get("Details", "")))

        items_html.append(
            f"""
            <div class="timeline-item type-{html.escape(event_type.lower())}">
                <div class="dot">{icon}</div>
                <div class="day">Day {day}</div>
                <div class="card">
                    <div class="card-top">
                        <div class="event-title">{event}</div>
                        <div class="event-type">{html.escape(event_type)}</div>
                    </div>
                    <div class="event-detail">{details}</div>
                </div>
            </div>
            """
        )

    timeline_height = min(900, max(360, 185 + len(timeline_df) * 92))

    components.html(
        f"""
        <style>
            body {{
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: transparent;
                color: #172033;
            }}

            .timeline-summary {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 10px;
                margin: 4px 0 18px 0;
            }}

            .mini {{
                background: #f8fafc;
                border: 1px solid #d9e2ef;
                border-radius: 16px;
                padding: 12px 14px;
                box-sizing: border-box;
            }}

            .mini-label {{
                color: #63708a;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: .05em;
                text-transform: uppercase;
            }}

            .mini-value {{
                color: #172033;
                font-size: 22px;
                font-weight: 900;
                margin-top: 4px;
            }}

            .timeline {{
                position: relative;
                padding: 4px 0 8px 0;
            }}

            .timeline::before {{
                content: "";
                position: absolute;
                top: 10px;
                bottom: 10px;
                left: 102px;
                width: 2px;
                background: linear-gradient(180deg, #1e5bff, #d9e2ef);
                opacity: .45;
                border-radius: 999px;
            }}

            .timeline-item {{
                position: relative;
                display: grid;
                grid-template-columns: 76px 34px 1fr;
                gap: 12px;
                align-items: center;
                margin: 0 0 14px 0;
            }}

            .day {{
                text-align: right;
                color: #63708a;
                font-size: 12px;
                font-weight: 800;
            }}

            .dot {{
                width: 34px;
                height: 34px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #eaf1ff;
                border: 1px solid #d9e2ef;
                z-index: 2;
                box-shadow: 0 6px 18px rgba(31,41,55,.07);
                font-size: 16px;
            }}

            .card {{
                background: #ffffff;
                border: 1px solid #d9e2ef;
                border-radius: 18px;
                padding: 13px 15px;
                box-shadow: 0 10px 24px rgba(31,41,55,.06);
                box-sizing: border-box;
            }}

            .card-top {{
                display: flex;
                justify-content: space-between;
                gap: 10px;
                align-items: center;
                margin-bottom: 5px;
            }}

            .event-title {{
                color: #172033;
                font-size: 15px;
                font-weight: 900;
                line-height: 1.25;
            }}

            .event-type {{
                color: #63708a;
                background: #f3f7fb;
                border: 1px solid #d9e2ef;
                font-size: 11px;
                font-weight: 800;
                padding: 4px 8px;
                border-radius: 999px;
                white-space: nowrap;
            }}

            .event-detail {{
                color: #5f6f89;
                font-size: 13px;
                line-height: 1.45;
                overflow-wrap: anywhere;
            }}

            .type-imaging .dot {{ background: #eef7ff; }}
            .type-treatment .dot {{ background: #fff4e8; }}
            .type-outcome .dot {{ background: #fff0f2; }}
            .type-diagnosis .dot {{ background: #eaf1ff; }}

            @media (max-width: 760px) {{
                .timeline-summary {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
                .timeline::before {{
                    left: 24px;
                }}
                .timeline-item {{
                    grid-template-columns: 34px 1fr;
                    gap: 10px;
                }}
                .day {{
                    grid-column: 2;
                    text-align: left;
                    margin-bottom: -8px;
                }}
                .dot {{
                    grid-row: span 2;
                }}
            }}
        </style>

        {summary_html}

        <div class="timeline">
            {''.join(items_html)}
        </div>
        """,
        height=timeline_height,
        scrolling=True,
    )

    with st.expander("Show timeline event table", expanded=False):
        st.dataframe(
            timeline_df[["Day", "Event", "Type", "Details"]],
            use_container_width=True,
            hide_index=True,
            height=220,
        )

    if not imaging_meta_df.empty:
        with st.expander("Show imaging visit details", expanded=False):
            st.dataframe(
                imaging_meta_df,
                use_container_width=True,
                hide_index=True,
                height=240,
            )


def render_info_card(label: str, value: str, help_text: str = "", icon: str = "📌") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-icon-layout">
                <div class="metric-icon">{icon}</div>
                <div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:22px; line-height:1.1;">{value}</div>
                    <div class="metric-help">{help_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clinical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Group"].str.startswith("Clinical") | df["Group"].isin(["Dataset / Cohort"])
    return df[mask].copy()


def genomic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Group"].isin(["Gene Expression", "Genomic / Biomarkers"])
    return df[mask].copy()


def imaging_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Group"].isin(["Imaging / Radiomics", "Clinical / Tumour Imaging Summary"])].copy()


def top_features_to_dataframe_from_response_text(text: str) -> pd.DataFrame:
    rows = []
    if not text:
        return pd.DataFrame(columns=["Feature", "Value", "Contribution"])

    pattern = re.compile(r"^-\s*(.*?)\s*=\s*(.*?)\s*\(contribution=([\-0-9.]+)\)", re.MULTILINE)
    for match in pattern.finditer(text):
        raw_feature = match.group(1).strip()
        value = match.group(2).strip()
        contribution = float(match.group(3))
        rows.append({
            "Feature": feature_metadata(raw_feature)["display_name"],
            "Raw Feature": raw_feature,
            "Value": value,
            "Contribution": contribution,
            "Abs Contribution": abs(contribution),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Abs Contribution", ascending=False)
    return out


def extract_probability_from_agent_text(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"Probability:\s*([0-9.]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_risk_from_agent_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Risk:\s*([^\n]+)", text)
    if not m:
        return None
    return m.group(1).strip()


def extract_ai_analysis(text: str) -> str:
    if not text:
        return ""
    if "AI Analysis:" in text:
        return text.split("AI Analysis:", 1)[-1].strip()
    return text.strip()


def call_agentcore_http(input_text: str, session_id: str, actor_id: str) -> Dict[str, Any]:
    payload = {
        "input": input_text,
        "session_id": session_id,
        "actor_id": actor_id,
    }
    try:
        response = requests.post(AGENTCORE_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "AgentCore request timed out."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to AgentCore backend."}
    except Exception as e:
        return {"status": "error", "message": f"AgentCore request failed: {str(e)}"}


def render_metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =======================================================
# Session state
# =======================================================
if "patient_id" not in st.session_state:
    st.session_state.patient_id = DEFAULT_PATIENT_ID
if "features" not in st.session_state:
    st.session_state.features = None
if "features_error" not in st.session_state:
    st.session_state.features_error = None
if "last_loaded_patient" not in st.session_state:
    st.session_state.last_loaded_patient = None
if "agent_response" not in st.session_state:
    st.session_state.agent_response = None
if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = str(uuid.uuid4())
if "actor_id" not in st.session_state:
    st.session_state.actor_id = DEFAULT_ACTOR_ID


# =======================================================
# Sidebar
# =======================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-row">
                <div class="sidebar-logo">🫁</div>
                <div>
                    <div class="sidebar-title">NSCLC AI</div>
                    <div class="sidebar-subtitle">Decision support dashboard</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="theme-row-note">Display mode</div>', unsafe_allow_html=True)
    theme = st.radio(
        "Theme",
        ["System", "Light", "Dark"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="compact-sidebar-title">👤 Patient</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="compact-sidebar-text">Enter the patient ID, or any patient information you have. The current demo loads cases by Patient ID.</div>',
        unsafe_allow_html=True,
    )

    st.session_state.patient_id = st.text_input(
        "Patient ID / MRN",
        value=st.session_state.patient_id,
        placeholder="e.g., R01-029",
        help="Enter the patient identifier used in the feature store and imaging viewer.",
    )

    auto_load = st.checkbox(
        "Auto-load",
        value=True,
        help="Loads patient data when the ID changes.",
    )

    load_clicked = st.button(
        "Load patient data",
        use_container_width=True,
        help="Fetch clinical, genomic, and imaging-derived model-related data.",
    )

    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="compact-sidebar-title">✨ AI Decision Support</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="compact-sidebar-text">Run this when you want AI risk assessment, key drivers, and a clinical interpretation for the selected patient.</div>',
        unsafe_allow_html=True,
    )

    run_ai_clicked = st.button(
        "Run AI analysis",
        use_container_width=True,
        help="Runs prediction, top drivers, and clinical explanation through AgentCore.",
    )

    with st.expander("Advanced", expanded=False):
        st.session_state.actor_id = st.text_input(
            "Clinician/session ID",
            value=st.session_state.actor_id,
        )
        st.caption(f"AgentCore endpoint: `{AGENTCORE_URL}`")
        st.caption(f"HealthImaging datastore: `{HEALTHIMAGING_DATASTORE_NAME}`")

inject_css(theme)


# =======================================================
# Header
# =======================================================
logo_path = get_cpi_logo_path(theme)
title_col, logo_col = st.columns([5.5, 1])

with title_col:
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">tiny NG-Dx</div>'
        '<div class="hero-subtitle">'
        'A lightweight demonstration of the NG-Dx multimodal radiogenomics platform, '
        'built on the public NSCLC Radiogenomics dataset from '
        '<a href="https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/" target="_blank">'
        'The Cancer Imaging Archive (TCIA)</a>. '
        'This demo integrates clinical variables, genomic markers, radiomics, '
        'and CT imaging review in a compact clinician-oriented dashboard.'
        '</div>'
        '<div style="margin-top:16px;">'
        '<span class="pill">🫁 NSCLC Radiogenomics</span>'
        '<span class="pill">🧬 Multimodal AI</span>'
        '<span class="pill">🖼️ CT Imaging</span>'
        '<span class="pill">📊 Clinical Review</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

with logo_col:
    if logo_path is not None:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.image(str(logo_path), width=120)

# =======================================================
# Feature loading logic
# =======================================================
patient_id = st.session_state.patient_id.strip()
should_load = False

if load_clicked:
    should_load = True
elif auto_load and st.session_state.features is None and patient_id:
    should_load = True
elif auto_load and patient_id and st.session_state.last_loaded_patient != patient_id:
    should_load = True

if should_load:
    st.session_state.agent_response = None
    st.session_state.focus_model_related_tab = False
    with st.spinner("Loading patient data..."):
        result = fetch_patient_features(patient_id)

    if result.get("status") != "ok":
        st.session_state.features = None
        st.session_state.features_error = result.get("message", "Failed to fetch patient features.")
    else:
        st.session_state.features = result.get("data", {}).get("features", {})
        st.session_state.features_error = None
        st.session_state.last_loaded_patient = patient_id


# =======================================================
# AI HTTP call logic
# =======================================================
if run_ai_clicked:
    if not patient_id:
        st.error("Patient ID is required before running AI interpretation.")
    else:
        with st.spinner("Generating AI interpretation through AgentCore..."):
            response = call_agentcore_http(
                input_text=patient_id,
                session_id=st.session_state.agent_session_id,
                actor_id=st.session_state.actor_id,
            )
        st.session_state.agent_response = response
        if response.get("status") == "ok":
            st.session_state.focus_ai_tab = True
            st.success("AI interpretation generated. Opening AI Interpretation first.")
        else:
            st.error(response.get("message", "AI interpretation failed."))


# =======================================================
# Empty / error states
# =======================================================
if st.session_state.features_error:
    st.error(st.session_state.features_error)
    st.stop()

if not st.session_state.features:
    st.info("No patient data loaded yet. Enter a Patient ID / MRN or keep automatic loading enabled.")
    st.stop()


# =======================================================
# Data preparation
# =======================================================
features = st.session_state.features
features_df = features_to_dataframe(features)
active_df = active_one_hot_summary(features_df)

agent_response = st.session_state.agent_response or {}
agent_text = agent_response.get("response", "") if agent_response.get("status") == "ok" else ""
agent_prob = extract_probability_from_agent_text(agent_text)
agent_risk = extract_risk_from_agent_text(agent_text)
top_driver_df = top_features_to_dataframe_from_response_text(agent_text)
ai_analysis = extract_ai_analysis(agent_text)

available_count = int((~features_df["Missing"]).sum())
missing_count = int(features_df["Missing"].sum())
numeric_count = int(features_df["Numeric Value"].notna().sum())




def safe_rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def render_ai_risk_card(label: str, value: str, help_text: str, icon: str = "🤖") -> None:
    st.markdown(
        f"""
        <div class="ai-risk-card">
            <div class="ai-risk-top">
                <div class="ai-risk-icon">{icon}</div>
                <div>
                    <div class="ai-risk-label">{label}</div>
                    <div class="ai-risk-value">{value}</div>
                </div>
            </div>
            <div class="ai-risk-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_risk_summary_block() -> None:
    if not (agent_risk or agent_prob is not None):
        return

    st.markdown("#### AI Risk Summary")
    r1, r2 = st.columns(2)
    with r1:
        risk_value = str(agent_risk or "N/A").title()
        icon = "🟢" if str(agent_risk or "").lower() == "low" else "🟠" if str(agent_risk or "").lower() in {"medium", "moderate"} else "🔴"
        render_ai_risk_card("Risk Level", risk_value, "Predicted risk group from the AI model.", icon)

    with r2:
        if agent_prob is not None:
            render_ai_risk_card(
                "Risk Probability",
                f"{agent_prob:.1%}",
                f"Raw model probability: {agent_prob:.3f}",
                "📊",
            )
            st.progress(min(max(agent_prob, 0), 1))
        else:
            render_ai_risk_card("Risk Probability", "N/A", "Probability was not found in the AI response.", "📊")

# =======================================================
# Tabs
# =======================================================
if st.session_state.get("focus_model_related_tab"):
    tab_model_inputs, tab_clinical_overview, tab_genomic, tab_viewer, tab_analysis, tab_raw = st.tabs(
        [
            "📥 Model-Related Data",
            "🩺 Clinical Overview",
            "🧬 Genomic Markers",
            "🖼️ Image View",
            "🤖 AI Interpretation",
            "🧾 Raw Data",
        ]
    )
elif st.session_state.get("focus_ai_tab"):
    tab_analysis, tab_clinical_overview, tab_genomic, tab_viewer, tab_model_inputs, tab_raw = st.tabs(
        [
            "🤖 AI Interpretation",
            "🩺 Clinical Overview",
            "🧬 Genomic Markers",
            "🖼️ Image View",
            "📥 Model-Related Data",
            "🧾 Raw Data",
        ]
    )
else:
    tab_clinical_overview, tab_genomic, tab_viewer, tab_analysis, tab_model_inputs, tab_raw = st.tabs(
        [
            "🩺 Clinical Overview",
            "🧬 Genomic Markers",
            "🖼️ Image View",
            "🤖 AI Interpretation",
            "📥 Model-Related Data",
            "🧾 Raw Data",
        ]
    )


with tab_clinical_overview:
    st.markdown("### 🩺 Clinical Overview")
    st.caption("Clinician-facing summary of the selected patient, with key demographics, clinical status, treatment, recurrence, and biomarker status in one place.")

    survival_label, survival_help = survival_status_label(features_df)

    overview_items = [
        ("Patient ID", patient_id, "Current case", "🪪"),
        ("Age", value_for(features_df, ["ageathistologicaldiagnosis"]), "At histological diagnosis", "📅"),
        ("Gender", active_value_for_prefix(features_df, "gender_"), "Recorded category", "⚧"),
        ("Ethnicity", active_value_for_prefix(features_df, "ethnicity_"), "Recorded category", "🌍"),
        ("Smoking Status", active_value_for_prefix(features_df, "smokingstatus_"), "Clinical history", "🚬"),
        ("Pack Years", value_for(features_df, ["packyears"]), "Smoking exposure", "📦"),
        ("Histology", active_value_for_prefix(features_df, "histology_"), "Tumour type", "🔬"),
        ("T Stage", active_value_for_prefix(features_df, "pathologicaltstage_"), "Pathological T stage", "🎯"),
        ("N Stage", active_value_for_prefix(features_df, "pathologicalnstage_"), "Pathological N stage", "🧩"),
        ("M Stage", active_value_for_prefix(features_df, "pathologicalmstage_"), "Pathological M stage", "🧭"),
        ("Survival Status", survival_label, survival_help, "💓"),
        ("Time to Death", value_for(features_df, ["timetodeathdays"]), "Days, if available", "⏱️"),
    ]

    for i in range(0, len(overview_items), 4):
        cols = st.columns(4)
        for col, (label, value, help_text, icon) in zip(cols, overview_items[i:i + 4]):
            with col:
                render_info_card(label, value, help_text, icon)

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Treatment and Recurrence")
        treatment_rows = [
            ("Adjuvant Treatment", active_value_for_prefix(features_df, "adjuvanttreatment_")),
            ("Chemotherapy", active_value_for_prefix(features_df, "chemotherapy_")),
            ("Radiation Therapy", active_value_for_prefix(features_df, "radiation_")),
            ("Recurrence", active_value_for_prefix(features_df, "recurrence_")),
            ("Recurrence Location", active_value_for_prefix(features_df, "recurrencelocation_")),
        ]
        treatment_df = pd.DataFrame(treatment_rows, columns=["Item", "Value"])
        st.dataframe(treatment_df, use_container_width=True, hide_index=True, height=230)

    with right:
        st.markdown("#### Key Biomarker Status")
        biomarker_rows = [
            ("EGFR", active_value_for_prefix(features_df, "egfrmutationstatus_")),
            ("KRAS", active_value_for_prefix(features_df, "krasmutationstatus_")),
            ("ALK", active_value_for_prefix(features_df, "alktranslocationstatus_")),
            ("Pleural Invasion", active_value_for_prefix(features_df, "pleuralinvasionelasticvisceralorparietal_")),
            ("Lymphovascular Invasion", active_value_for_prefix(features_df, "lymphovascularinvasion_")),
        ]
        biomarker_summary_df = pd.DataFrame(biomarker_rows, columns=["Item", "Value"])
        st.dataframe(biomarker_summary_df, use_container_width=True, hide_index=True, height=230)


    st.divider()
    render_clinical_timeline(features_df, patient_id)

    st.divider()
    st.markdown("#### Full Clinical Details")
    st.caption("Searchable clinical fields. Inactive one-hot columns stay hidden unless requested, because apparently models enjoy making simple categories into many tiny columns.")

    clinical_df = clinical_dataframe(features_df)
    active_clinical = clinical_df[(clinical_df["Active"] == "Yes") | (~clinical_df["Active"].isin(["Yes", "No"]))].copy()

    c1, c2 = st.columns([1, 2])
    with c1:
        show_all_clinical = st.checkbox("Show inactive one-hot fields", value=False)
    with c2:
        clinical_search = st.text_input("Search clinical details", placeholder="stage, smoking, survival, histology...")

    clinical_view = clinical_df if show_all_clinical else active_clinical
    if clinical_search:
        s = clinical_search.lower()
        clinical_view = clinical_view[
            clinical_view["Display Name"].str.lower().str.contains(s, na=False)
            | clinical_view["Raw Feature"].str.lower().str.contains(s, na=False)
        ]

    st.dataframe(
        clinical_view[["Display Name", "Value", "Active", "Group", "Subgroup", "Raw Feature"]],
        use_container_width=True,
        height=480,
        hide_index=True,
    )




with tab_genomic:
    st.markdown("### 🧬 Genomic Markers")
    st.caption("Mutation/translocation status and gene-expression signature for the selected patient.")

    genomic_df = genomic_dataframe(features_df)
    biomarker_df = genomic_df[genomic_df["Group"] == "Genomic / Biomarkers"].copy()
    gene_df = genomic_df[genomic_df["Group"] == "Gene Expression"].copy()

    # -----------------------------
    # Biomarker summary cards
    # -----------------------------
    st.markdown("#### Biomarker Snapshot")
    biomarker_cards = [
        ("EGFR", active_value_for_prefix(features_df, "egfrmutationstatus_"), "Mutation status", "🧬"),
        ("KRAS", active_value_for_prefix(features_df, "krasmutationstatus_"), "Mutation status", "🧪"),
        ("ALK", active_value_for_prefix(features_df, "alktranslocationstatus_"), "Translocation status", "🔁"),
    ]

    bcols = st.columns(3)
    for col, (label, value, help_text, icon) in zip(bcols, biomarker_cards):
        with col:
            render_info_card(label, value, help_text, icon)

    if not biomarker_df.empty:
        with st.expander("Show biomarker feature table", expanded=False):
            biomarker_active = biomarker_df[
                (biomarker_df["Active"] == "Yes")
                | (~biomarker_df["Active"].isin(["Yes", "No"]))
            ].copy()
            st.dataframe(
                biomarker_active[["Display Name", "Value", "Active", "Subgroup", "Raw Feature"]],
                use_container_width=True,
                height=220,
                hide_index=True,
            )

    st.divider()

    # -----------------------------
    # Gene-expression visual summary
    # -----------------------------
    st.markdown("#### Gene Expression Profile")
    heat = gene_df.dropna(subset=["Numeric Value"])[["Display Name", "Numeric Value", "Raw Feature"]].copy()

    if heat.empty:
        st.info("No numeric gene-expression values available.")
    else:
        heat["Numeric Value"] = pd.to_numeric(heat["Numeric Value"], errors="coerce")
        heat = heat.dropna(subset=["Numeric Value"]).sort_values("Numeric Value", ascending=False)

        mean_expr = float(heat["Numeric Value"].mean())
        max_gene = str(heat.iloc[0]["Display Name"])
        max_val = float(heat.iloc[0]["Numeric Value"])
        min_gene = str(heat.iloc[-1]["Display Name"])
        min_val = float(heat.iloc[-1]["Numeric Value"])

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_info_card("Genes", str(len(heat)), "Expression features", "🧫")
        with s2:
            render_info_card("Mean Expression", f"{mean_expr:.3g}", "Across signature", "📈")
        with s3:
            render_info_card("Highest Gene", max_gene, f"Value: {max_val:.3g}", "⬆️")
        with s4:
            render_info_card("Lowest Gene", min_gene, f"Value: {min_val:.3g}", "⬇️")

        chart_left, chart_right = st.columns([1.25, 1])

        with chart_left:
            heat_matrix = pd.DataFrame([heat["Numeric Value"].values], columns=heat["Display Name"].values)
            fig = px.imshow(
                heat_matrix,
                aspect="auto",
                labels=dict(x="Gene", y="Patient", color="Expression"),
                title="Expression Heatmap",
            )
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=48, b=10),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig_bar = px.bar(
                heat.sort_values("Numeric Value", ascending=True),
                x="Numeric Value",
                y="Display Name",
                orientation="h",
                title="Ranked Gene Expression",
            )
            fig_bar.update_layout(height=520, margin=dict(l=10, r=10, t=48, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_right:
            top_n = min(10, len(heat))
            top_genes = heat.head(top_n).copy()
            fig_polar = px.line_polar(
                top_genes,
                r="Numeric Value",
                theta="Display Name",
                line_close=True,
                title=f"Top {top_n} Gene Expression Radar",
            )
            fig_polar.update_traces(fill="toself")
            fig_polar.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_polar, use_container_width=True)

            bottom_genes = heat.tail(top_n).sort_values("Numeric Value", ascending=True).copy()
            fig_low = px.bar(
                bottom_genes,
                x="Numeric Value",
                y="Display Name",
                orientation="h",
                title=f"Lowest {top_n} Expressed Genes",
            )
            fig_low.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_low, use_container_width=True)

        with st.expander("Show gene-expression table", expanded=False):
            st.dataframe(
                gene_df[["Display Name", "Value", "Group", "Subgroup", "Raw Feature"]],
                use_container_width=True,
                height=360,
                hide_index=True,
            )



with tab_viewer:
    st.markdown("### 🖼️ Image View")
    st.caption("CT thumbnail preview and quick access to the OHIF viewer.")

    url = build_ohif_patient_url(patient_id)

    top_left, top_right = st.columns([2.2, 1])
    with top_left:
        st.markdown("#### Patient Imaging")
        st.caption(f"Patient ID / MRN: {patient_id}")
    with top_right:
        st.link_button(
            "Open Full Study in OHIF",
            url,
            use_container_width=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    img_col, info_col = st.columns([1.05, 1.95])

    with img_col:
        st.markdown("<div class='image-thumb-wrap'>", unsafe_allow_html=True)
        try:
            thumbnail_bytes = fetch_thumbnail_from_s3(patient_id)

            if thumbnail_bytes:
                render_streamlit_image(
                    thumbnail_bytes,
                    caption=f"CT thumbnail for patient {patient_id}",
                )
            else:
                st.info("No CT thumbnail available for this patient.")

        except Exception as e:
            st.warning(f"Could not load thumbnail from S3: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown(
            """
            <div class="image-action-panel">
                <div class="image-action-title">Full imaging study</div>
                <div class="image-action-text">
                    Use OHIF for CT review, series navigation, and image-level inspection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



with tab_model_inputs:
    st.markdown("### 📥 Model-Related Data")
    st.caption("All numeric and categorical variables prepared for the model, including clinical data, genomic markers, and imaging-derived features. Human-readable names are added for clinical and technical review.")

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">For the model</div>
            <div class="panel-subtitle">
                This section combines the previous <b>Model-Related Data</b> and <b>Imaging Features</b> tabs.
                Use it to inspect every feature passed to the prediction pipeline, including radiomics families and raw feature names.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if agent_response.get("status") == "ok":
        st.markdown("#### AI Risk Output")
        render_ai_risk_summary_block()

        if not top_driver_df.empty:
            st.markdown("##### SHAP-style Driver Contribution")
            st.caption("Positive values push the predicted risk upward; negative values push it downward. These are parsed from the AI model driver output.")

            shap_view = top_driver_df.sort_values("Contribution", ascending=True).copy()
            fig_shap = px.bar(
                shap_view,
                x="Contribution",
                y="Feature",
                orientation="h",
                title="SHAP-style Feature Contributions",
            )
            fig_shap.add_vline(x=0, line_width=1, line_dash="dash")
            fig_shap.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("Show AI risk driver table", expanded=False):
                st.dataframe(top_driver_df, use_container_width=True, height=320, hide_index=True)
        else:
            st.info("No SHAP-style driver output was found in the AI response.")

        st.divider()

    imaging_df = imaging_dataframe(features_df)
    if not imaging_df.empty:
        with st.expander("Show imaging-derived feature families", expanded=False):
            family_counts = imaging_df.groupby("Subgroup").size().reset_index(name="Count")
            fig_img = px.bar(
                family_counts.sort_values("Count", ascending=True),
                x="Count",
                y="Subgroup",
                orientation="h",
                title="Imaging Feature Families",
            )
            fig_img.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_img, use_container_width=True)


    with st.expander("Show model-input distribution charts", expanded=False):
        c_chart1, c_chart2 = st.columns([1, 1])
        with c_chart1:
            group_counts = features_df.groupby("Group").size().reset_index(name="Count")
            fig = px.bar(
                group_counts.sort_values("Count", ascending=True),
                x="Count",
                y="Group",
                orientation="h",
                title="Model Input Sections",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c_chart2:
            subgroup_counts = features_df.groupby("Subgroup").size().reset_index(name="Count")
            fig2 = px.treemap(
                subgroup_counts,
                path=["Subgroup"],
                values="Count",
                title="Model Input Subgroups",
            )
            fig2.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)


    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    with c1:
        selected_groups = st.multiselect(
            "Data section",
            options=sorted(features_df["Group"].dropna().unique()),
            default=sorted(features_df["Group"].dropna().unique()),
        )
    with c2:
        active_filter = st.selectbox("Filter", ["All", "Selected categories only", "Non-zero numeric", "Missing only"])
    with c3:
        search = st.text_input("Search", placeholder="Search display name or raw feature name...")

    filtered = features_df[features_df["Group"].isin(selected_groups)].copy()

    if active_filter == "Selected categories only":
        filtered = filtered[filtered["Active"] == "Yes"]
    elif active_filter == "Non-zero numeric":
        filtered = filtered[filtered["Numeric Value"].fillna(0) != 0]
    elif active_filter == "Missing only":
        filtered = filtered[filtered["Missing"]]

    if search:
        s = search.lower()
        filtered = filtered[
            filtered["Display Name"].str.lower().str.contains(s, na=False)
            | filtered["Raw Feature"].str.lower().str.contains(s, na=False)
        ]

    visible_cols = ["Display Name", "Value", "Active", "Group", "Subgroup", "Raw Feature", "Missing"]
    st.dataframe(filtered[visible_cols], use_container_width=True, height=620, hide_index=True)

    st.download_button(
        "Download filtered table",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_features_filtered.csv",
        mime="text/csv",
    )



with tab_analysis:
    st.markdown("### 🤖 AI Interpretation")
    if not ai_analysis:
        st.info("Run AI Decision Support to view the clinical summary, risk estimate, and top drivers.")
    else:
        render_ai_risk_summary_block()

        st.markdown("#### Clinical Explanation")
        st.markdown(ai_analysis)

        st.divider()
        st.markdown("#### Model Output")
        st.caption("You can view the model results there. Click the button below.")

        if st.button("Open Model-Related Data → AI Risk Output", use_container_width=False):
            st.session_state.focus_model_related_tab = True
            safe_rerun()


with tab_raw:
    st.markdown("### 🧾 Raw Data")
    st.caption("Raw values are preserved for technical review. Display names are added only to make the table survivable by humans.")
    st.json(features)

    st.download_button(
        "Download raw features",
        data=json.dumps(features, indent=2, default=str).encode("utf-8"),
        file_name=f"{patient_id}_raw_features.json",
        mime="application/json",
    )

    if agent_response:
        st.markdown("### Raw AgentCore Response")
        st.json(agent_response)
        st.download_button(
            "Download AI response",
            data=json.dumps(agent_response, indent=2, default=str).encode("utf-8"),
            file_name=f"{patient_id}_agentcore_response.json",
            mime="application/json",
        )
