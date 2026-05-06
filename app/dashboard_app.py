import os
import re
import sys
import json
import uuid
from textwrap import dedent
from pathlib import Path
from urllib.parse import quote_plus
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
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

THUMBNAIL_BUCKET = os.getenv(
    "THUMBNAIL_BUCKET",
    "nsclc-medical-image-data-811165582441-eu-west-2-an",
)

THUMBNAIL_PREFIX = os.getenv(
    "THUMBNAIL_PREFIX",
    "processed/nsclc_radiogenomics/PNG",
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
            border-radius: 20px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(16,24,40,.06);
            min-height: 112px;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 12px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: .055em;
        }}

        .metric-value {{
            font-size: 27px;
            font-weight: 950;
            letter-spacing: -.04em;
            margin-top: 8px;
            line-height: 1.05;
        }}

        .metric-help {{
            color: var(--muted);
            font-size: 12px;
            margin-top: 6px;
            line-height: 1.35;
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


def render_info_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:22px; line-height:1.1;">{value}</div>
            <div class="metric-help">{help_text}</div>
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
                    <div class="sidebar-title">NSCLC Patient Review</div>
                    <div class="sidebar-subtitle">Clinical dashboard and imaging access</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">Display</div>', unsafe_allow_html=True)
    theme = st.radio(
        "Theme",
        ["System", "Light", "Dark"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Patient lookup</div></div>', unsafe_allow_html=True)
    st.session_state.patient_id = st.text_input(
        "Patient ID / MRN",
        value=st.session_state.patient_id,
        placeholder="R01-029",
        help="Enter the patient identifier used in the feature store and imaging viewer.",
    )

    auto_load = st.checkbox(
        "Load patient data automatically",
        value=True,
        help="Loads patient features as soon as the ID changes.",
    )

    load_clicked = st.button(
        "Load patient data",
        use_container_width=True,
        help="Fetch clinical, genomic, and imaging-derived model inputs.",
    )

    st.markdown(
        """
        <div class="sidebar-help">
            Patient data loads first. The AI interpretation is separate so the dashboard opens quickly.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">AI interpretation</div></div>', unsafe_allow_html=True)
    run_ai_clicked = st.button(
        "Generate AI interpretation",
        use_container_width=True,
        help="Runs prediction, top drivers, and clinical explanation through AgentCore.",
    )

    with st.expander("Advanced settings", expanded=False):
        st.session_state.actor_id = st.text_input(
            "Clinician/session ID",
            value=st.session_state.actor_id,
        )
        st.caption(f"AgentCore endpoint: `{AGENTCORE_URL}`")

    st.markdown(
        """
        <div class="sidebar-status">
            <span class="status-dot"></span>
            Feature-first mode active
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css(theme)


# =======================================================
# Header
# =======================================================
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
    unsafe_allow_html=True
)

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
            st.success("AI interpretation generated.")
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


# =======================================================
# Top cards
# =======================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Patient", patient_id, "Current loaded case")
with col2:
    render_metric_card("Patient Data", str(len(features_df)), f"{available_count} available, {missing_count} missing")
with col3:
    render_metric_card("Model Inputs", str(numeric_count), "Numeric values prepared for the model")
with col4:
    ai_status = "Generated" if agent_response.get("status") == "ok" else "Not generated"
    render_metric_card("AI Interpretation", ai_status, "Run on demand")


# =======================================================
# Risk summary if AI is available
# =======================================================
if agent_response.get("status") == "ok":
    st.markdown(
        '<div class="panel"><div class="panel-title">AI Risk Summary</div><div class="panel-subtitle">Generated after clicking “Generate AI interpretation”. Initial patient review stays fast.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        render_metric_card("Risk Group", str(agent_risk or "N/A"), "AI model output")
    with c2:
        if agent_prob is not None:
            render_metric_card("Risk Probability", f"{agent_prob:.3f}", f"{agent_prob:.1%}")
        else:
            render_metric_card("Risk Probability", "N/A", "Not found in response")
    with c3:
        if agent_prob is not None:
            st.progress(min(max(agent_prob, 0), 1))
        st.markdown(f"[Open patient images in OHIF]({build_ohif_patient_url(patient_id)})")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">AI Risk Summary</div>
            <div class="panel-subtitle">
                Click <b>Generate AI interpretation</b> in the sidebar when you want prediction,
                top drivers, and clinical explanation. Until then, this page only loads patient data.
                Civilisation briefly wins against loading spinners.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =======================================================
# Tabs
# =======================================================
tab_overview, tab_clinical, tab_genomic, tab_imaging, tab_viewer, tab_features, tab_drivers, tab_analysis, tab_raw = st.tabs(
    [
        "Patient Overview",
        "Clinical Details",
        "Genomic Markers",
        "Imaging Features",
        "Image Viewer",
        "All Model Inputs",
        "AI Risk Drivers",
        "AI Interpretation",
        "Raw Data",
    ]
)


with tab_overview:
    st.markdown("### Patient Overview")
    st.caption("A clinician-facing summary of the selected patient. Technical model-input charts are now under All Model Inputs.")

    survival_label, survival_help = survival_status_label(features_df)

    overview_items = [
        ("Patient ID", patient_id, "Current case"),
        ("Age", value_for(features_df, ["ageathistologicaldiagnosis"]), "At histological diagnosis"),
        ("Gender", active_value_for_prefix(features_df, "gender_"), "Recorded category"),
        ("Ethnicity", active_value_for_prefix(features_df, "ethnicity_"), "Recorded category"),
        ("Smoking Status", active_value_for_prefix(features_df, "smokingstatus_"), "Clinical history"),
        ("Pack Years", value_for(features_df, ["packyears"]), "Smoking exposure"),
        ("Histology", active_value_for_prefix(features_df, "histology_"), "Tumour type"),
        ("T Stage", active_value_for_prefix(features_df, "pathologicaltstage_"), "Pathological T stage"),
        ("N Stage", active_value_for_prefix(features_df, "pathologicalnstage_"), "Pathological N stage"),
        ("M Stage", active_value_for_prefix(features_df, "pathologicalmstage_"), "Pathological M stage"),
        ("Survival Status", survival_label, survival_help),
        ("Time to Death", value_for(features_df, ["timetodeathdays"]), "Days, if available"),
    ]

    for i in range(0, len(overview_items), 4):
        cols = st.columns(4)
        for col, (label, value, help_text) in zip(cols, overview_items[i:i + 4]):
            with col:
                render_info_card(label, value, help_text)

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

    url = build_ohif_patient_url(patient_id)
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Imaging Review</div>
            <div class="panel-subtitle">
                Open the imaging viewer for this patient when reviewing radiology context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("Open patient images in OHIF", url, use_container_width=False)


with tab_clinical:
    st.markdown("### Clinical Details")
    st.caption("Key patient information and observed outcome are shown first. The full clinical table is below.")

    cards = clinical_key_cards(features_df)
    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (label, value, help_text) in zip(cols, cards[i:i + 4]):
            with col:
                render_info_card(label, value, help_text)

    st.divider()
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
    st.markdown("### Genomic Markers")
    st.caption("Mutation/translocation status and gene-expression signature for the selected patient.")

    genomic_df = genomic_dataframe(features_df)
    biomarker_df = genomic_df[genomic_df["Group"] == "Genomic / Biomarkers"].copy()
    gene_df = genomic_df[genomic_df["Group"] == "Gene Expression"].copy()

    if not biomarker_df.empty:
        st.markdown("#### Biomarker Status")
        biomarker_active = biomarker_df[(biomarker_df["Active"] == "Yes") | (~biomarker_df["Active"].isin(["Yes", "No"]))]
        st.dataframe(
            biomarker_active[["Display Name", "Value", "Active", "Subgroup", "Raw Feature"]],
            use_container_width=True,
            height=220,
            hide_index=True,
        )

    if not gene_df.empty:
        st.markdown("#### Gene Expression Signature")
        heat = gene_df.dropna(subset=["Numeric Value"])[["Display Name", "Numeric Value"]].copy()
        if heat.empty:
            st.info("No numeric gene-expression values available.")
        else:
            heat = heat.sort_values("Numeric Value", ascending=False)
            heat_matrix = pd.DataFrame([heat["Numeric Value"].values], columns=heat["Display Name"].values)
            fig = px.imshow(
                heat_matrix,
                aspect="auto",
                labels=dict(x="Gene", y="Patient", color="Expression"),
                title="Gene Expression Heatmap",
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            fig_bar = px.bar(
                heat.sort_values("Numeric Value", ascending=True),
                x="Numeric Value",
                y="Display Name",
                orientation="h",
                title="Gene Expression Values",
            )
            fig_bar.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            gene_df[["Display Name", "Value", "Group", "Subgroup", "Raw Feature"]],
            use_container_width=True,
            height=360,
            hide_index=True,
        )


with tab_imaging:
    st.markdown("### Imaging-Derived Features")
    st.caption("Radiomics and tumour imaging summary features. Patient images are opened from the Image Viewer tab.")

    imaging_df = imaging_dataframe(features_df)
    if imaging_df.empty:
        st.info("No imaging-derived features found.")
    else:
        family_counts = imaging_df.groupby("Subgroup").size().reset_index(name="Count")
        fig = px.bar(
            family_counts.sort_values("Count", ascending=True),
            x="Count",
            y="Subgroup",
            orientation="h",
            title="Imaging Feature Families",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns([1.2, 2])
        with c1:
            imaging_subgroups = st.multiselect(
                "Imaging feature family",
                options=sorted(imaging_df["Subgroup"].dropna().unique()),
                default=sorted(imaging_df["Subgroup"].dropna().unique()),
            )
        with c2:
            imaging_search = st.text_input("Search imaging features", placeholder="GLCM, shape, entropy, volume...")

        imaging_view = imaging_df[imaging_df["Subgroup"].isin(imaging_subgroups)].copy()
        if imaging_search:
            s = imaging_search.lower()
            imaging_view = imaging_view[
                imaging_view["Display Name"].str.lower().str.contains(s, na=False)
                | imaging_view["Raw Feature"].str.lower().str.contains(s, na=False)
            ]

        st.dataframe(
            imaging_view[["Display Name", "Value", "Group", "Subgroup", "Raw Feature"]],
            use_container_width=True,
            height=560,
            hide_index=True,
        )


with tab_viewer:
    st.markdown("### Imaging Review")
    st.caption("CT thumbnail preview and quick access to the OHIF viewer.")

    url = build_ohif_patient_url(patient_id)

    st.markdown("### Patient Imaging")
    st.caption(f"Patient ID / MRN: {patient_id}")

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

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.link_button(
        "Open Full Study in OHIF",
        url,
        use_container_width=True,
    )



with tab_features:
    st.markdown("### All Model Inputs")
    st.caption("Search and filter the complete feature set used by the model. Raw names are kept for debugging.")

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


with tab_drivers:
    st.markdown("### AI Risk Drivers")
    if top_driver_df.empty:
        st.info("Generate AI interpretation to see the main factors influencing the risk estimate.")
    else:
        st.dataframe(top_driver_df, use_container_width=True, height=320, hide_index=True)
        fig = px.bar(
            top_driver_df.sort_values("Contribution"),
            x="Contribution",
            y="Feature",
            orientation="h",
            title="Main AI Risk Drivers",
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)


with tab_analysis:
    st.markdown("### AI Interpretation")
    if not ai_analysis:
        st.info("Generate AI interpretation to view the clinical summary.")
    else:
        st.markdown(ai_analysis)


with tab_raw:
    st.markdown("### Raw Data")
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
