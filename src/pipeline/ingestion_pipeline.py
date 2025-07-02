import yaml
from src.components.data_ingestion import ingest_csv_to_sqlite
from src.components.normalizer import normalize_tweets

def run():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["new_raw_csv_path"]
    db_path = config["data"]["sqlite_db_path"]
    raw_table = config["data"]["new_raw_table"]
    interaction_table = config["data"]["new_interaction_table"]
    resolved_csv_path = config.get("output", {}).get("resolved_chat_log_csv", None)

    ingest_csv_to_sqlite(
        csv_path=csv_path,
        db_path=db_path,
        table_name=raw_table,
        resolved_csv_path=resolved_csv_path
    )

    normalize_tweets(
        db_path=db_path,
        raw_table=raw_table,
        output_table=interaction_table
    )
    print("✅ Ingestion pipeline complete.")