"""
Retention Knowledge Base
Contains e-commerce retention strategies and FAISS indexing logic for RAG.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Telecom Retention Strategies KB ---
RETENTION_STRATEGIES = [
    {
        "category": "Network & Quality",
        "content": "For customers in low-tier cities or reporting network issues: Offer a technical audit of their local cell tower performance. Provide a temporary 'Network Performance' credit or speed boost."
    },
    {
        "category": "Price & Competition",
        "content": "For customers with high monthly charges or sensitivity to price: Offer a tailored 'Loyalty Data Bundle' or a move to a more cost-effective contract with extra value (e.g., free streaming subscription)."
    },
    {
        "category": "Usage & Engagement",
        "content": "For users with low data/voice usage: Send personalized tutorials on 5G benefits or offer a 1-month trial of a premium service (e.g., cloud storage, security suite) to increase stickiness."
    },
    {
        "category": "Contract Renewal",
        "content": "For customers nearing the end of their tenure: Proactively offer an early handset upgrade or a 'Contract Renewal Bonus' data allowance to prevent switching to competitors."
    },
    {
        "category": "Customer Support",
        "content": "For customers with high complaints: Assign a dedicated 'Priority Support Agent' for 3 months and offer a sincere apology with a bill discount as a gesture of goodwill."
    }
]

def get_retention_vectorstore():
    """Build or load the FAISS vector store for retention strategies."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    docs = [
        Document(page_content=s["content"], metadata={"category": s["category"]})
        for s in RETENTION_STRATEGIES
    ]
    
    # In-memory FAISS for simplicity, but can be persisted
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
