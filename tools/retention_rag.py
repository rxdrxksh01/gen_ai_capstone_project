"""
Tool: retrieve_retention_strategies
Performs RAG to find e-commerce best practices for specific customer risks.
"""

from langchain_core.tools import tool
from src.retention_kb import get_retention_vectorstore

# Cache the vectorstore to avoid re-initializing on every call
vectorstore = None

@tool
def retrieve_retention_strategies(query: str) -> str:
    """Search for e-commerce retention strategies based on customer analysis.
    
    Args:
        query: A string describing the customer's top risks or drivers (e.g., 'low satisfaction, high churn').
        
    Returns:
        A formatted string of relevant retention strategies retrieved from the knowledge base.
    """
    global vectorstore
    if vectorstore is None:
        vectorstore = get_retention_vectorstore()
    
    results = vectorstore.similarity_search(query, k=2)
    
    formatted_results = "\n\n".join([
        f"--- Strategy ({doc.metadata['category']}) ---\n{doc.page_content}"
        for doc in results
    ])
    
    return formatted_results
