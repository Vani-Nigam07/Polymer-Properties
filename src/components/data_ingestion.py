import pandas as pd
import os
from tqdm import tqdm
from src.utils.db_handler import get_connection, create_new_raw_table


def ingest_csv_to_sqlite(csv_path, db_path, table_name, resolved_csv_path=None, max_rows=2000, chunk_size=1000):
    conn = get_connection(db_path)
    create_new_raw_table(conn, table_name)

    resolved_customers = set()
    if resolved_csv_path and os.path.exists(resolved_csv_path):
        try:
            resolved_df = pd.read_csv(resolved_csv_path)
            resolved_customers = set(
                resolved_df[resolved_df["issue_status"] == "resolved"]["customer_id"].astype(str)
            )
            print(f"✔ Loaded {len(resolved_customers)} resolved customers from chat log.")
        except Exception as e:
            print(f"⚠ Failed to read resolved chat log: {e}")

    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    cursor = conn.cursor()
    current_rows = 0

    def insert_chunk(df_chunk):
        rows = []
        for _, row in df_chunk.iterrows():
            customer_id = str(row['author_id'])
            if str(row['inbound']) == "1" and customer_id in resolved_customers:
                continue  # Skip resolved inbound messages

            rows.append((
                str(row['tweet_id']),
                str(row['author_id']),
                bool(row['inbound']),
                str(row['created_at']),
                str(row['text']),
                str(row['response_tweet_id']) if not pd.isna(row['response_tweet_id']) else None,
                str(row['in_response_to_tweet_id']) if not pd.isna(row['in_response_to_tweet_id']) else None
            ))

        if rows:
            cursor.executemany(f"""
                INSERT OR IGNORE INTO {table_name}
                (tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, rows)
            conn.commit()

    for chunk in tqdm(reader, desc="Ingesting chunks"):
        insert_chunk(chunk)
        current_rows += len(chunk)
        if current_rows >= max_rows:
            break

    conn.close()
