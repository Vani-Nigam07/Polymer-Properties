import numpy as np
import pandas as pd
from dateutil import parser

FINE_TO_HIGH_MAP = {
    "technical issue": "Technical Issue", "bug or glitch": "Technical Issue",
    "login problem": "Technical Issue", "app crash or freeze": "Technical Issue",
    "slow performance": "Technical Issue", "update failure": "Technical Issue",
    "account issue": "Account & Payment", "billing problem": "Account & Payment",
    "payment failure": "Account & Payment", "subscription issue": "Account & Payment",
    "unauthorized charge": "Account & Payment",
    "delivery problem": "Delivery & Orders", "order not received": "Delivery & Orders",
    "damaged item": "Delivery & Orders", "wrong item received": "Delivery & Orders",
    "missing package": "Delivery & Orders",
    "product question": "Product & Services", "pricing": "Product & Services",
    "service unavailable": "Product & Services", "feature request": "Product & Services",
    "refund": "Customer Experience", "complaint": "Customer Experience",
    "feedback": "Customer Experience", "praise": "Customer Experience",
    "cancellation": "Customer Experience"
}

def get_sentiment_scores(convo, sentiment_model):
    for turn in convo["conversation"]:
        if turn["sender"] == "user":
            result = sentiment_model(turn["text"][:1000])[0]
            turn["sentiment"] = {
                "label": result["label"].lower(),
                "confidence": float(result["score"])
            }

def get_issue_tags(convo, zero_shot_model):
    full_text = " ".join([t["text"] for t in convo["conversation"]])
    result = zero_shot_model(full_text, list(FINE_TO_HIGH_MAP.keys()))
    fine = result["labels"][0]
    return {
        "fine_grained": fine,
        "high_level": FINE_TO_HIGH_MAP.get(fine, "Miscellaneous"),
        "confidence": float(result["scores"][0])
    }

def classify_trajectory(sentiments):
    if not sentiments or len(sentiments) < 2:
        return "stable"
    scores = [1 if s == "positive" else -1 for s in sentiments]
    if scores[-1] > scores[0]:
        return "improving"
    elif scores[-1] < scores[0]:
        return "worsening"
    elif all(s == scores[0] for s in scores):
        return "stable"
    return "mixed"

def compute_avg_response_gap(convo):
    timestamps = convo["conversation"]
    timestamps.sort(key=lambda x: x["timestamp"])
    gaps = []
    for i in range(len(timestamps) - 1):
        a, b = timestamps[i], timestamps[i + 1]
        if a["sender"] == "user" and b["sender"] == "agent":
            try:
                t1 = parser.parse(a["timestamp"])
                t2 = parser.parse(b["timestamp"])
                gaps.append(abs((t2 - t1).total_seconds()))
            except:
                continue
    return float(np.mean(gaps)) if gaps else 0.0

def summarize_conversations(flows):
    for convo in flows:
        user_turns = [t for t in convo["conversation"] if t["sender"] == "user"]
        agent_turns = [t for t in convo["conversation"] if t["sender"] == "agent"]
        
        sentiments = [t["sentiment"]["label"] for t in user_turns if "sentiment" in t]
        confidences = [t["sentiment"]["confidence"] for t in user_turns if "sentiment" in t]

        convo["trajectory_category"] = classify_trajectory(sentiments)
        convo["flow_summary"] = {
            "num_user_turns": len(user_turns),
            "num_agent_turns": len(agent_turns),
            "avg_sentiment_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "median_sentiment_confidence": float(np.median(confidences)) if confidences else 0.0,
            "avg_response_gap_seconds": compute_avg_response_gap(convo)
        }

def cluster_conversations(flows, csv_path, embedding_model):
    from bertopic import BERTopic

    texts = [
        " ".join([t["text"] for t in convo["conversation"] if t["sender"] == "user"])[:2000]
        for convo in flows
    ]
    customer_ids = [convo["customer_id"] for convo in flows]

    topic_model = BERTopic(embedding_model=embedding_model)
    topics, _ = topic_model.fit_transform(texts)

    df_clusters = pd.DataFrame({
        "customer_id": customer_ids,
        "flow_cluster": topics
    })
    df_clusters.to_csv(csv_path, index=False)

    for convo, topic in zip(flows, topics):
        convo["flow_cluster"] = int(topic)

def tag_conversations(flows, sentiment_model, zero_shot_model):
    tagged = []
    for convo in flows:
        get_sentiment_scores(convo, sentiment_model)
        convo["issue_type"] = get_issue_tags(convo, zero_shot_model)
        tagged.append(convo)
    return tagged
