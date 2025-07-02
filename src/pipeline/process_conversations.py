# Updated process_conversations.py

import sqlite3, yaml, json, pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.models.loaders import load_sentiment_model, load_zero_shot_model, load_embedding_model
from src.components.conversation_parser import build_conversation_flows
from src.components.tagger import tag_conversations, summarize_conversations, cluster_conversations

def run():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    db_path = config['data']['sqlite_db_path']
    interaction_table = config['sqlite']['new_interaction_table']
    parsed_path = config['output']['parsed_conversations_path']
    tagged_path = config['output']['tagged_conversations_path']
    cluster_csv_path = config['output']['flow_clusters_csv']

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {interaction_table};", conn)
    conn.close()

    if Path(parsed_path).exists():
        with open(parsed_path, 'r', encoding='utf-8') as f:
            old_convos = json.load(f)
        existing_ids = {d['customer_id'] for d in old_convos}
    else:
        old_convos = []
        existing_ids = set()

    new_df = df[~df['customer_id'].isin(existing_ids)].copy()
    new_flows = build_conversation_flows(new_df)
    all_flows = old_convos + new_flows

    with open(parsed_path, 'w', encoding='utf-8') as f:
        json.dump(all_flows, f, indent=2, ensure_ascii=False)

    sentiment_model = load_sentiment_model()
    zero_shot_model = load_zero_shot_model()

    tagged_flows = tag_conversations(new_flows, sentiment_model, zero_shot_model)
    summarize_conversations(tagged_flows)
    cluster_conversations(tagged_flows, cluster_csv_path, load_embedding_model())

    if Path(tagged_path).exists():
        with open(tagged_path, 'r', encoding='utf-8') as f:
            previous = json.load(f)
    else:
        previous = []

    final_flows = previous + tagged_flows
    with open(tagged_path, 'w', encoding='utf-8') as f:
        json.dump(final_flows, f, indent=2, ensure_ascii=False)

    print(f"✅ User behavior analysis pipeline complete → {tagged_path}")