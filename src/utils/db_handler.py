import sqlite3

def get_connection(db_path):
    return sqlite3.connect(db_path)

def create_new_raw_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            tweet_id TEXT PRIMARY KEY,
            author_id TEXT,
            inbound BOOLEAN,
            created_at TEXT,
            text TEXT,
            response_tweet_id TEXT,
            in_response_to_tweet_id TEXT
        );
    """)
    conn.commit()
