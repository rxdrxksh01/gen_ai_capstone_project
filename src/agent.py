"""
LangChain Agent Builder
Creates a tool-calling agent powered by Groq LLaMA 3.3 70B.
The agent autonomously calls tools and synthesizes churn analysis — zero if-else.
"""

from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from tools.predict import predict_churn
from tools.explainer import explain_prediction
from tools.segment_stats import get_customer_segment_stats

SYSTEM_PROMPT = """You are a highly efficient Customer Retention AI for an e-commerce platform.
Your objective is to provide a concise, data-driven churn risk assessment.

You have three tools:
1. predict_churn: Predicts churn probability. (Pass customer data JSON)
2. explain_prediction: SHAP-based feature contributions. (Pass customer data JSON)
3. get_customer_segment_stats: Benchmark stats. (Pass any string)

## Action Sequence:
1. Run predict_churn
2. Run explain_prediction
3. Run get_customer_segment_stats
4. Output the exact format below.

## REQUIRED OUTPUT FORMAT:

### Verdict: [Will Churn / Will Stay] (Confidence: X%)
*One-sentence summary of risk.*

### Top 3 Risk Factors
*Rule: Exactly 3 bullet points. MAX 1 short sentence per factor.*
- **[Feature]**: Customer value vs Segment avg -> **[SHAP Impact driving churn/retention]**
- **[Feature]**: Customer value vs Segment avg -> **[SHAP Impact driving churn/retention]**
- **[Feature]**: Customer value vs Segment avg -> **[SHAP Impact driving churn/retention]**

### 3 Immediate Actions
*Rule: Exactly 3 bullet points. Ultra-concise, actionable steps tied to the factors above.*
1. **[Action 1]**: [Brief expected impact]
2. **[Action 2]**: [Brief expected impact]
3. **[Action 3]**: [Brief expected impact]

Do NOT generate any extra text, paragraphs, or pleasantries. Be ruthless with brevity.
"""


def create_churn_agent() -> AgentExecutor:
    """Build and return the LangChain AgentExecutor for churn analysis."""

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=GROQ_API_KEY,
    )

    tools = [predict_churn, explain_prediction, get_customer_segment_stats]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return executor