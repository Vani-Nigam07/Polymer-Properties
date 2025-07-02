import json
import pandas as pd
from tqdm import tqdm
import sqlite3
from models.llm_loader import load_llama_model
from src.components.nba_prompt import nba_prompt_template, output_parser
from src.components.nba_engine import generate_nba_instruction
from src.components.normalizer import normalize_tweets
import yaml

def run():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    nba_json_path = config["output"]["tagged_conversations_path"]
    nba_chatlog_csv = config["output"]["resolved_chat_log_csv"]
    nba_updated_json = config["output"]["nba_updated_json"]
    sqlite_path = config["data"]["sqlite_db_path"]
    new_raw_table = config["sqlite"]["new_raw_table"]
    new_interaction_table = config["sqlite"]["new_interaction_table"]

    with open(nba_json_path, "r", encoding="utf-8") as f:
        all_conversations = json.load(f)

    llm = load_llama_model()

    nba_outputs, nba_records = [], []
    for convo in tqdm(all_conversations):
        try:
            updated_convo, record = generate_nba_instruction(convo, llm, nba_prompt_template, output_parser)
            nba_outputs.append(updated_convo)
            nba_records.append(record)
        except Exception as e:
            print(f"⚠️ Error for customer {convo.get('customer_id', 'unknown')}: {e}")

    with open(nba_updated_json, "w", encoding="utf-8") as f:
        json.dump(nba_outputs, f, indent=2)

    pd.DataFrame(nba_records).to_csv(nba_chatlog_csv, index=False)
    print("✅ NBA outputs and chat log saved.")

    print("📤 Injecting NBA agent replies back into raw tweets table...")
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    agent_rows = []
    for record in nba_records:
        customer_id = str(record["customer_id"])
        tweet_id = f"agent_{customer_id}_{record['send_time']}"
        author_id = "AmazonHelp"
        inbound = False
        created_at = record["send_time"]
        text = record["message"]
        response_tweet_id = None
        in_response_to_tweet_id = customer_id

        agent_rows.append((
            tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
        ))

    cursor.executemany(f"""
        INSERT OR IGNORE INTO {new_raw_table} 
        (tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, agent_rows)
    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(agent_rows)} agent replies into `{new_raw_table}`")

    print("🔄 Re-running normalization to update interaction table...")
    normalize_tweets(
        db_path=sqlite_path,
        raw_table=new_raw_table,
        output_table=new_interaction_table
    )
    print("🎯 NBA pipeline complete: replies injected + normalized.")