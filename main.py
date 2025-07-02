from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import yaml
import os
import uuid

# Import the pipelines directly
from src.pipeline import ingestion_pipeline, process_conversations, nba_pipeline

app = FastAPI(title="Customer Support NBA System")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

csv_path = config["data"]["new_raw_csv_path"]

# Ensure CSV exists
if not os.path.exists(csv_path):
    pd.DataFrame(columns=[
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id"
    ]).to_csv(csv_path, index=False)

class TweetInput(BaseModel):
    author_id: str
    text: str
    in_response_to_tweet_id: str = "NA"
    response_tweet_id: str = "NA"

@app.post("/submit_tweet/")
def submit_tweet(tweet: TweetInput):
    new_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().strftime("%a %b %d %H:%M:%S +0000 %Y")

    row = {
        "tweet_id": new_id,
        "author_id": tweet.author_id,
        "inbound": True,
        "created_at": created_at,
        "text": tweet.text,
        "response_tweet_id": tweet.response_tweet_id,
        "in_response_to_tweet_id": tweet.in_response_to_tweet_id
    }

    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    # Directly run all three pipelines
    ingestion_pipeline.run()
    process_conversations.run()
    nba_pipeline.run()

    return {
        "message": "Tweet submitted and pipelines executed successfully.",
        "data": row
    }
