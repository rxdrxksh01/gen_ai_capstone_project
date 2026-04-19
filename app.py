"""
Streamlit App — Customer Churn Predictor with AI Agent
Powered by LangChain + Groq LLaMA 3.3 70B + SHAP Explainability.
Zero if-else for churn logic — everything is handled by the LLM agent.
"""

import json
import streamlit as st
from src.agent import create_churn_agent

# ─── Page Config ───
st.set_page_config(
    page_title="Telecom Churn Predictor — AI Agent",
    layout="wide",
)

# ─── Cache the agent so it's not recreated on every rerun ───
@st.cache_resource
def get_agent():
    return create_churn_agent()

agent_executor = get_agent()

# ─── Header ───
st.title("Telecom Churn Predictor")
st.caption(
    "Powered by **LangChain Agent** · Groq LLaMA 3.3 70B · Telecom Network Guard"
)
st.markdown("---")

# ─── Session State Defaults ───
default_values = {
    "Tenure": 0,
    "CityTier": 1,
    "WarehouseToHome": 0.0,
    "HourSpendOnApp": 0.0,
    "NumberOfDeviceRegistered": 0.0,
    "SatisfactionScore": 3,
    "NumberOfAddress": 0.0,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 0.0,
    "CouponUsed": 0.0,
    "OrderCount": 0.0,
    "DaySinceLastOrder": 0.0,
    "CashbackAmount": 0.0,
    "PreferredLoginDevice": "Mobile Phone",
    "PreferredPaymentMode": "UPI",
    "Gender": "Male",
    "PreferedOrderCat": "Grocery",
    "MaritalStatus": "Single",
    "messages": [],
    "last_context": {},
}

for key, value in default_values.items():
    st.session_state.setdefault(key, value)


# ─── Custom CSS for Premium UI ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Lora:ital,wght@0,500;1,500&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Main Background & Text */
    .stApp {
        background-color: #FAFAF7; /* Warm off-white */
        color: #3C3D37; /* Dark earthy grey/brown */
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Lora', serif;
        color: #2D3339; /* Deep Slate */
        font-weight: 500;
        letter-spacing: -0.2px;
    }

    /* Streamlit Buttons */
    .stButton>button {
        background: #6B7F8C; /* Slate / River Stone */
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 4px rgba(107, 127, 140, 0.2);
    }
    .stButton>button:hover {
        background: #556873;
        color: #FFFFFF;
        box-shadow: 0 4px 8px rgba(85, 104, 115, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton>button:active {
        background: #3E4D57;
        color: #FFFFFF;
    }

    /* Analysis Container */
    .analysis-card {
        background: #FFFFFF;
        border: 1px solid #E8E6E1;
        border-radius: 12px;
        padding: 32px;
        box-shadow: 0 8px 30px rgba(45, 51, 57, 0.05);
        margin-top: 1rem;
        color: #3C3D37;
    }
    .analysis-card hr {
        border-color: #E8E6E1;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F3F2EB; /* Soft sandstone */
        border-right: 1px solid #E8E6E1;
    }
    
    /* Sidebar headers */
    .sidebar-title {
        color: #79868F; /* Muted slate */
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 1px solid #E8E6E1;
        padding-bottom: 5px;
    }
    
    /* Input Fields override for light mode */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #FFFFFF !important;
        color: #3C3D37 !important;
        border: 1px solid #DFDDD3 !important;
        border-radius: 6px;
    }
    
    .stSlider>div>div>div>div {
        background-color: #6B7F8C !important; /* Slate slider */
    }
    
    /* Fix markdown text color */
    .element-container markdown, p {
        color: #3C3D37 !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #ffffff;
        color: #3C3D37;
    }
</style>
""", unsafe_allow_html=True)

# ─── Layout ───

# Configure Sidebar for Inputs
with st.sidebar:
    st.markdown('<div class="sidebar-title">Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "High Risk",
            use_container_width=True,
            on_click=lambda: st.session_state.update({
                "Tenure": 0, "CityTier": 3, "WarehouseToHome": 60.0,
                "HourSpendOnApp": 0.0, "NumberOfDeviceRegistered": 1.0,
                "SatisfactionScore": 1, "NumberOfAddress": 1.0, "Complain": 1,
                "OrderAmountHikeFromlastYear": 0.0, "CouponUsed": 0.0,
                "OrderCount": 0.0, "DaySinceLastOrder": 180.0, "CashbackAmount": 0.0,
                "PreferredLoginDevice": "Mobile Phone",
                "PreferredPaymentMode": "Cash on Delivery",
                "Gender": "Male", "PreferedOrderCat": "Mobile",
                "MaritalStatus": "Single",
            }),
        )

    with col2:
        st.button(
            "Low Risk",
            use_container_width=True,
            on_click=lambda: st.session_state.update({
                "Tenure": 24, "CityTier": 1, "WarehouseToHome": 5.0,
                "HourSpendOnApp": 5.0, "NumberOfDeviceRegistered": 3.0,
                "SatisfactionScore": 5, "NumberOfAddress": 3.0, "Complain": 0,
                "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 5.0,
                "OrderCount": 20.0, "DaySinceLastOrder": 5.0, "CashbackAmount": 500.0,
                "PreferredLoginDevice": "Mobile Phone",
                "PreferredPaymentMode": "UPI",
                "Gender": "Female", "PreferedOrderCat": "Fiber",
                "MaritalStatus": "Married",
            }),
        )
        
    st.markdown("---")
    st.markdown('<div class="sidebar-title">Customer Profile</div>', unsafe_allow_html=True)
    
    st.session_state["Tenure"] = st.number_input("Tenure (months)", min_value=0, value=st.session_state["Tenure"])
    st.session_state["SatisfactionScore"] = st.slider("Satisfaction Score", 1, 5, value=st.session_state["SatisfactionScore"])
    st.session_state["Complain"] = st.selectbox("Filed Complaint?", [0, 1], index=[0, 1].index(st.session_state["Complain"]))
    st.session_state["CashbackAmount"] = st.number_input("Cashback Amount", min_value=0.0, value=st.session_state["CashbackAmount"])
    st.session_state["DaySinceLastOrder"] = st.number_input("Days Since Last Order", min_value=0.0, value=st.session_state["DaySinceLastOrder"])

    with st.expander("Show All Customer Details"):
        st.session_state["CityTier"] = st.number_input("City Tier", min_value=1, max_value=3, value=st.session_state["CityTier"])
        st.session_state["WarehouseToHome"] = st.number_input("Warehouse → Home", min_value=0.0, value=st.session_state["WarehouseToHome"])
        st.session_state["HourSpendOnApp"] = st.number_input("Hours Spent On App", min_value=0.0, value=st.session_state["HourSpendOnApp"])
        st.session_state["NumberOfDeviceRegistered"] = st.number_input("Devices Registered", min_value=0.0, value=st.session_state["NumberOfDeviceRegistered"])
        st.session_state["NumberOfAddress"] = st.number_input("Number Of Addresses", min_value=0.0, value=st.session_state["NumberOfAddress"])
        st.session_state["OrderAmountHikeFromlastYear"] = st.number_input("Order Amt Hike", value=st.session_state["OrderAmountHikeFromlastYear"])
        st.session_state["CouponUsed"] = st.number_input("Coupons Used", min_value=0.0, value=st.session_state["CouponUsed"])
        st.session_state["OrderCount"] = st.number_input("Order Count", min_value=0.0, value=st.session_state["OrderCount"])
        
        st.session_state["PreferredLoginDevice"] = st.selectbox("Login Device", ["Mobile Phone", "Computer"], index=["Mobile Phone", "Computer"].index(st.session_state["PreferredLoginDevice"]))
        st.session_state["PreferredPaymentMode"] = st.selectbox("Payment Mode", ["UPI", "Credit Card", "Debit Card", "Cash on Delivery", "E wallet", "COD"], index=["UPI", "Credit Card", "Debit Card", "Cash on Delivery", "E wallet", "COD"].index(st.session_state["PreferredPaymentMode"]))
        st.session_state["Gender"] = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state["Gender"]))
        
        # Safety check for Telecom categories
        service_options = ["Internet", "Fiber", "Mobile", "Streaming", "Others"]
        current_cat = st.session_state.get("PreferedOrderCat", "Internet")
        if current_cat not in service_options:
            current_cat = "Internet"
            
        st.session_state["PreferedOrderCat"] = st.selectbox("Service Category", service_options, index=service_options.index(current_cat))
        st.session_state["MaritalStatus"] = st.selectbox("Marital Status", ["Single", "Married"], index=["Single", "Married"].index(st.session_state["MaritalStatus"]))

# Main Area
st.title("Telecom Retention Intelligence: Agentic AI Churn Strategy")
st.markdown("Deep network usage & behavioural analysis powered by **Groq LLaMA 70B**.")

# Use session state to persist the agent result across reruns
st.session_state.setdefault("agent_result", None)

def _execute_agent():
    """Invoke the LangChain agent with the current customer data."""
    customer_data = {k: v for k, v in st.session_state.items() if k in default_values}
    customer_json = json.dumps(customer_data)

    with st.spinner("LangGraph Agent reasoning and retrieving strategies..."):
        response = agent_executor.invoke(
            {"input": customer_json}
        )

    # Store full context for chat follow-ups
    st.session_state["agent_result"] = response["output"]
    st.session_state["last_context"] = {
        "prediction": response["prediction"],
        "explanation": response["explanation"],
        "strategies": response["strategies"]
    }
    st.session_state["messages"] = [] # Reset chat when new data is analyzed
    _render_results()

def _render_results():
    has_result = bool(st.session_state["agent_result"])
    
    def _show_card():
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown(st.session_state["agent_result"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Chat Section ---
        st.markdown("---")
        st.subheader("💬 Strategy Consultation")
        st.caption("Ask specific questions about this customer's risk or the recommended actions.")
        
        # Display messages
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat Input
        if prompt := st.chat_input("Ex: 'Draft a personalized apology email for this customer'"):
            # Add user message
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate and add AI message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = agent_executor.chat(
                        prompt, 
                        st.session_state["last_context"],
                        st.session_state["messages"][:-1]
                    )
                    st.markdown(response)
                    st.session_state["messages"].append({"role": "assistant", "content": response})

    def _show_info():
        st.info("Configure the customer profile in the sidebar and hit **Analyze Customer Risk**.")
        
    display_map = {True: _show_card, False: _show_info}
    display_map[has_result]()

run_agent = st.button("Analyze Customer Risk", type="primary")

# ─── Dispatch: dict-lookup pattern instead of if-else ───
_dispatch = {True: _execute_agent, False: _render_results}
_dispatch[run_agent]()

