# src/components/nba_engine.py
from datetime import datetime, timedelta, timezone
from dateutil import parser
import copy

def generate_nba_instruction(convo, llm, prompt_template, output_parser):
    summary = convo.get("flow_summary", {})
    original_messages = copy.deepcopy(convo.get("conversation", []))

    conversation_text = "\n".join([
        f"{m['sender'].capitalize()}: {m['text']}" for m in original_messages if 'text' in m
    ])

    prompt_input = {
        "customer_id": convo["customer_id"],
        "agent_id": convo["agent_id"],
        "num_user_turns": summary.get("num_user_turns", 0),
        "num_agent_turns": summary.get("num_agent_turns", 0),
        "avg_sentiment_confidence": summary.get("avg_sentiment_confidence", 0.0),
        "median_sentiment_confidence": summary.get("median_sentiment_confidence", 0.0),
        "avg_response_gap_seconds": summary.get("avg_response_gap_seconds", 0.0),
        "fine_grained_issue": summary.get("fine_grained_issue", ""),
        "high_level_issue": summary.get("high_level_issue", ""),
        "trajectory_category": summary.get("trajectory_category", "stable"),
        "flow_cluster": convo.get("flow_cluster", -1),
        "conversation_text": conversation_text
    }

    prompt = prompt_template.format(**prompt_input)
    response = llm.invoke(prompt)
    content = getattr(response, "content", "")

    parsed = output_parser.parse(content)

    # Time fix
    now = datetime.now(timezone.utc)
    try:
        send_time = parser.parse(parsed["send_time"])
        if send_time.tzinfo is None:
            send_time = send_time.replace(tzinfo=timezone.utc)
    except:
        send_time = now + timedelta(hours=1)

    send_time = min(send_time, now + timedelta(days=2)).replace(
        year=now.year, month=now.month, day=min(now.day + 2, 28)
    )
    parsed["send_time"] = send_time.isoformat().replace("+00:00", "Z")

    # Append agent message
    convo["conversation"].append({
        "sender": "agent",
        "text": parsed["message"],
        "timestamp": parsed["send_time"]
    })

    # Chat log (original convo only)
    chat_log = "\n".join([
        f"{m['sender'].capitalize()}: {m['text']}" for m in original_messages
    ])

    nba_log = {
        "customer_id": convo["customer_id"],
        "chat_log": chat_log,
        "channel": parsed["channel"],
        "message": parsed["message"],
        "send_time": parsed["send_time"],
        "reasoning": parsed["reasoning"],
        "issue_status": "resolved" if parsed["resolved"] else "pending_customer_response"
    }

    return convo, nba_log
