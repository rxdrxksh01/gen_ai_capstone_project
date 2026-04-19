"""
LangGraph Agent — Milestone 2 Refactor
Implements a structured state machine with RAG for e-commerce retention.
"""

import json
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage

from src.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from tools.predict import predict_churn
from tools.explainer import explain_prediction
from tools.retention_rag import retrieve_retention_strategies

# ─── 1. Define State ───
class AgentState(TypedDict):
    input: str
    customer_data: dict
    prediction: dict
    explanation: dict
    strategies: str
    final_output: str

# ─── 2. Define Nodes ───

def predict_node(state: AgentState):
    """Run churn prediction logic."""
    # Extra data from input string (assuming JSON string passed as input)
    # or passed directly in state. For our app, we'll pass customer_json in input.
    customer_json = state["input"]
    result_str = predict_churn.invoke(customer_json)
    return {"prediction": json.loads(result_str), "customer_data": json.loads(customer_json)}

def explain_node(state: AgentState):
    """Run SHAP explanation logic."""
    customer_json = json.dumps(state["customer_data"])
    result_str = explain_prediction.invoke(customer_json)
    return {"explanation": json.loads(result_str)}

def rag_node(state: AgentState):
    """RAG Step: Retrieve retention strategies based on findings."""
    prediction = state["prediction"]
    explanation = state["explanation"]
    
    # Construct a query for the vector store
    top_feature = explanation["feature_contributions"][0]["feature"] if explanation["feature_contributions"] else "loyalty"
    query = f"Churn risk: {prediction['label']}, Top driver: {top_feature}"
    
    strategies = retrieve_retention_strategies.invoke(query)
    return {"strategies": strategies}

def synthesis_node(state: AgentState):
    """Synthesize final structured report using LLM."""
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=GROQ_API_KEY,
    )
    
    prompt = f"""You are a Customer Retention AI. Synthesize the following data into a structured report.

DATA:
- Prediction: {state['prediction']}
- SHAP Details: {state['explanation']}
- RAG Strategies: {state['strategies']}

REQUIRED FORMAT:
### Verdict: [Will Churn / Will Stay] (Confidence: X%)
*One-sentence summary of risk.*

### [HEADER: 'Top 3 Risk Factors' if churning, 'Top 3 Retention Drivers' if staying]
- **[Factor 1]**: [Detail]
- **[Factor 2]**: [Detail]
- **[Factor 3]**: [Detail]

### 3 Immediate Actions
*Rule: Use the retrieved RAG strategies to provide concrete, actionable steps.*
1. [Action]
2. [Action]
3. [Action]

Do NOT include pleasantries. Be concise.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_output": response.content}

# ─── 3. Build Graph ───

def create_churn_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("predict", predict_node)
    workflow.add_node("explain", explain_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("synthesis", synthesis_node)
    
    workflow.set_entry_point("predict")
    workflow.add_edge("predict", "explain")
    workflow.add_edge("explain", "rag")
    workflow.add_edge("rag", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

# For backwards compatibility with app.py (wrapped in an interface)
class LangGraphAgent:
    def __init__(self):
        self.graph = create_churn_graph()
        
    def invoke(self, inputs: dict) -> dict:
        # Input looks like {"input": "..."}
        result = self.graph.invoke(inputs)
        return {"output": result["final_output"]}

def create_churn_agent():
    """Factory function matching previous signature."""
    return LangGraphAgent()