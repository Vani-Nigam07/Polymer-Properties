# normalizer.py

import sqlite3
import pandas as pd
from src.utils.text_cleaner import clean_text

def normalize_tweets(db_path: str, raw_table: str, output_table: str):
    conn = sqlite3.connect(db_path)

    df = pd.read_sql(f"SELECT * FROM {raw_table}", conn)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['text_cleaned'] = df['text'].apply(clean_text)

    df = df[df['tweet_id'].notnull()]
    df['tweet_id'] = df['tweet_id'].astype(float).astype(int).astype(str)
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna(-1)
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].astype(float).astype(int).astype(str)

    inbound_df = df[df['inbound'] == 1].copy()
    outbound_df = df[df['inbound'] == 0].copy()

    merged = outbound_df.merge(
        inbound_df,
        left_on='in_response_to_tweet_id',
        right_on='tweet_id',
        suffixes=('_agent', '_cust')
    )

    merged['conversation_id'] = merged['tweet_id_cust']

    interaction_df = merged[[ 
        'conversation_id', 'author_id_cust', 'author_id_agent',
        'text_cleaned_cust', 'text_cleaned_agent', 'created_at_cust'
    ]].rename(columns={
        'author_id_cust': 'customer_id',
        'author_id_agent': 'agent_id',
        'text_cleaned_cust': 'customer_text',
        'text_cleaned_agent': 'agent_response',
        'created_at_cust': 'created_at'
    })

    try:
        existing = pd.read_sql(
            f"SELECT conversation_id, customer_id, agent_id, created_at FROM {output_table}",
            conn
        )
        interaction_df = interaction_df.merge(
            existing,
            on=['conversation_id', 'customer_id', 'agent_id', 'created_at'],
            how='left',
            indicator=True
        ).query('_merge == "left_only"').drop(columns=['_merge'])
    except:
        pass

    if not interaction_df.empty:
        interaction_df.to_sql(output_table, conn, index=False, if_exists='append')

    conn.close()
    print(f"Created interaction table with {len(interaction_df)} new rows.")
