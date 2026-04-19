"""
Retention Knowledge Base
Contains e-commerce retention strategies and FAISS indexing logic for RAG.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Sample Retention Strategies KB ---
RETENTION_STRATEGIES = [
    {
        "category": "High Churn Risk",
        "content": "For customers with high churn probability: Offer a significant 'We Miss You' discount (20%+) and prioritize their support tickets. Use multi-channel outreach (Email + Push Notification)."
    },
    {
        "category": "Customer Satisfaction",
        "content": "For low satisfaction scores (1-2): Trigger a proactive customer success call or a personalized apology video. Offer a 'satisfaction guarantee' refund or credit for their next order."
    },
    {
        "category": "Pricing & Cashback",
        "content": "For price-sensitive customers (low cashback/hike sensitivity): Enroll them in a tiered loyalty program where cashback increases with frequency. Highlight daily deals and bundles."
    },
    {
        "category": "Engagement",
        "content": "For low app engagement: Implement gamified rewards (daily login streaks) and send personalized product recommendations based on past browsing history."
    },
    {
        "category": "Delivery & Logistics",
        "content": "For customers far from the warehouse: Offer 'Express Shipping' discounts or provide more accurate real-time tracking to reduce delivery anxiety."
    },
    {
        "category": "Tenure / Loyalty",
        "content": "For long-term customers (high tenure): Send an 'Anniversary' gift or early access to new product launches to recognize their loyalty."
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
