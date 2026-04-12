import boto3
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup

# Config
REGION = "eu-west-2"

GENOMIC_FG = "genomic-feature-group-05-19-10-59"
CLINICAL_FG = "clinical-feature-group-05-18-48-56"
IMAGING_FG = "ct-seg-image-imaging-feature-group"

BUCKET = "multimodal-lung-cancer-811165582441-eu-west-2-an"
PREFIX = "multi-model-health-ml"

OUTPUT = f"s3://{BUCKET}/{PREFIX}/athena-results/"


# Session
def create_session():
    boto_session = boto3.Session(region_name=REGION)
    return Session(boto_session=boto_session)


# Query Runner
def run_query(query, sql):
    print("\n🧠 Running query:")
    print(sql)

    query.run(query_string=sql, output_location=OUTPUT)
    query.wait()

    print("Execution ID:", query._current_query_execution_id)

    df = query.as_dataframe()
    return df


# Inspect Feature Group
def inspect_feature_group(name, session):
    print(f"\n🔍 Inspecting: {name}")

    fg = FeatureGroup(name=name, sagemaker_session=session)
    desc = fg.describe()

    key_raw = desc["RecordIdentifierFeatureName"]
    key = key_raw.lower()

    print("Primary Key (FG):", key_raw)
    print("Primary Key (Athena):", key)
    print("Event Time:", desc["EventTimeFeatureName"])

    query = fg.athena_query()

    print("Athena Table:", query.table_name)

    sql = f"""
        SELECT *
        FROM "{query.table_name}"
        LIMIT 5
    """

    df = run_query(query, sql)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample rows:")
    print(df.head())


# Check key values
def check_key_values(name, session):
    print(f"\n🔑 Checking key values for: {name}")

    fg = FeatureGroup(name=name, sagemaker_session=session)
    desc = fg.describe()

    key_raw = desc["RecordIdentifierFeatureName"]
    key = key_raw.lower()

    query = fg.athena_query()

    sql = f"""
        SELECT DISTINCT {key}
        FROM "{query.table_name}"
        LIMIT 20
        """

    df = run_query(query, sql)

    print("\nSample Key Values:")
    print(df)


# Main
def run():
    session = create_session()

    for fg_name in [GENOMIC_FG, CLINICAL_FG, IMAGING_FG]:
        inspect_feature_group(fg_name, session)
        check_key_values(fg_name, session)


if __name__ == "__main__":
    run()