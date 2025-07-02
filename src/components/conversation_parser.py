import pandas as pd

def build_conversation_flows(df: pd.DataFrame):
    grouped = df.sort_values(by='created_at').groupby('customer_id')
    conversation_flows = []

    for customer_id, group in grouped:
        conversation = []
        seen = set()
        agent_id = group['agent_id'].mode().values[0] if not group['agent_id'].isnull().all() else None

        for _, row in group.iterrows():
            user_msg = ('user', row['customer_text'], row['created_at'])
            agent_msg = ('agent', row['agent_response'], row['created_at'])

            if user_msg not in seen:
                conversation.append({
                    "sender": user_msg[0],
                    "text": user_msg[1],
                    "timestamp": user_msg[2]
                })
                seen.add(user_msg)

            if agent_msg not in seen:
                conversation.append({
                    "sender": agent_msg[0],
                    "text": agent_msg[1],
                    "timestamp": agent_msg[2]
                })
                seen.add(agent_msg)

        conversation_flows.append({
            "customer_id": customer_id,
            "agent_id": agent_id,
            "conversation": conversation
        })

    return conversation_flows
