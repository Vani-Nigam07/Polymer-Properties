# src/components/nba_prompt.py
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="customer_id", description="Customer ID"),
    ResponseSchema(name="channel", description="Best communication channel"),
    ResponseSchema(name="send_time", description="Best send time in ISO format (UTC)"),
    ResponseSchema(name="message", description="Response message to send"),
    ResponseSchema(name="reasoning", description="Rationale behind channel/time/message"),
    ResponseSchema(name="resolved", description="True if issue is resolved")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

nba_prompt_template = PromptTemplate(
    template="""
You are an intelligent reasoning agent for Next Best Action (NBA) in customer support.

Your task is to recommend:
1. Best channel (twitter_dm_reply, scheduling_phone_call, or email_reply)
2. Best send time in UTC (consider customer activity & urgency)
3. A helpful, personalized response message
4. Reasoning behind your decision
5. Whether the issue appears resolved

---

You are provided a customer conversation flow with detailed behavior and analysis:

### Features Available

**trajectory_category** → whether user's mood is improving, worsening, or mixed/stable
**Conversation Summary:**
- Customer ID: {customer_id}
- Agent ID: {agent_id}
- Total User Turns: {num_user_turns}
- Total Agent Turns: {num_agent_turns}
- Average Sentiment Confidence: {avg_sentiment_confidence:.2f}
- Median Sentiment Confidence: {median_sentiment_confidence:.2f}
- Avg Response Gap (seconds): {avg_response_gap_seconds:.2f}
- Fine-Grained Issue: {fine_grained_issue}
- High-Level Issue: {high_level_issue}
- Sentiment Trajectory Category: {trajectory_category}
- Flow Cluster ID: {flow_cluster}

**Conversation Text Snippets**:
{conversation_text}

---

### Instructions

- Choose the most appropriate **channel**:
  - Use `twitter_dm_reply` if conversation is live and active
  - Use `scheduling_phone_call` for high-stakes/confusing/urgent problems
  - Use `email_reply` if sentiment has stabilized, or issue is resolved

- Pick best **send_time** in UTC (based on timestamps or urgency)

- Compose a helpful **message** (solve or clarify)

- Provide clear **reasoning**

- Set `resolved = true` only if user’s last message shows satisfaction

---

{format_instructions}
""",
    input_variables=[
        "customer_id", "agent_id", "num_user_turns", "num_agent_turns",
        "avg_sentiment_confidence", "median_sentiment_confidence",
        "avg_response_gap_seconds", "fine_grained_issue", "high_level_issue",
        "trajectory_category", "flow_cluster", "conversation_text"
    ],
    partial_variables={"format_instructions": format_instructions}
)
