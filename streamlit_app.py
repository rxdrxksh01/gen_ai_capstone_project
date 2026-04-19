import streamlit as st
import pandas as pd
import joblib

from src.config import PIPELINE_PATH

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load(PIPELINE_PATH)

model = load_model()

st.title("📊 Customer Churn Predictor")

# -----------------------------
# Initialize Session State Defaults
# -----------------------------
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
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Demo Buttons
# -----------------------------
st.markdown("### 🔎 Try Demo Customers")

col1, col2 = st.columns(2)

with col1:
    if st.button("Demo: High Churn Risk"):
        st.session_state.update({
            "Tenure": 0,
            "CityTier": 3,
            "WarehouseToHome": 60,
            "HourSpendOnApp": 0,
            "NumberOfDeviceRegistered": 1,
            "SatisfactionScore": 1,
            "NumberOfAddress": 1,
            "Complain": 1,
            "OrderAmountHikeFromlastYear": 0,
            "CouponUsed": 0,
            "OrderCount": 0,
            "DaySinceLastOrder": 180,
            "CashbackAmount": 0,
            "PreferredLoginDevice": "Mobile Phone",
            "PreferredPaymentMode": "Cash on Delivery",
            "Gender": "Male",
            "PreferedOrderCat": "Mobile",
            "MaritalStatus": "Single"
        })

with col2:
    if st.button("Demo: Low Churn Risk"):
        st.session_state.update({
            "Tenure": 24,
            "CityTier": 1,
            "WarehouseToHome": 5,
            "HourSpendOnApp": 5,
            "NumberOfDeviceRegistered": 3,
            "SatisfactionScore": 5,
            "NumberOfAddress": 3,
            "Complain": 0,
            "OrderAmountHikeFromlastYear": 15,
            "CouponUsed": 5,
            "OrderCount": 20,
            "DaySinceLastOrder": 5,
            "CashbackAmount": 500,
            "PreferredLoginDevice": "Mobile Phone",
            "PreferredPaymentMode": "UPI",
            "Gender": "Female",
            "PreferedOrderCat": "Grocery",
            "MaritalStatus": "Married"
        })


# -----------------------------
# Input Fields
# -----------------------------
st.markdown("### 📝 Enter Customer Details")

tenure = st.number_input("Tenure", min_value=0, key="Tenure")
city_tier = st.number_input("City Tier", min_value=1, key="CityTier")
warehouse = st.number_input("Warehouse To Home Distance", key="WarehouseToHome")
hour_spend = st.number_input("Hours Spent On App", key="HourSpendOnApp")
device = st.number_input("Number Of Devices Registered", key="NumberOfDeviceRegistered")
satisfaction = st.slider("Satisfaction Score", 1, 5, key="SatisfactionScore")
address = st.number_input("Number Of Addresses", key="NumberOfAddress")
complain = st.selectbox("Complain", [0, 1], key="Complain")
hike = st.number_input("Order Amount Hike From Last Year", key="OrderAmountHikeFromlastYear")
coupon = st.number_input("Coupon Used", key="CouponUsed")
order_count = st.number_input("Order Count", key="OrderCount")
last_order = st.number_input("Days Since Last Order", key="DaySinceLastOrder")
cashback = st.number_input("Cashback Amount", key="CashbackAmount")

login_device = st.selectbox(
    "Preferred Login Device",
    ["Mobile Phone", "Computer"],
    key="PreferredLoginDevice"
)

payment_mode = st.selectbox(
    "Preferred Payment Mode",
    ["UPI", "Credit Card", "Debit Card",
     "Cash on Delivery", "E wallet", "COD"],
    key="PreferredPaymentMode"
)

gender = st.selectbox("Gender", ["Male", "Female"], key="Gender")

order_category = st.selectbox(
    "Preferred Order Category",
    ["Grocery", "Mobile", "Laptop & Accessory",
     "Mobile Phone", "Others"],
    key="PreferedOrderCat"
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married"],
    key="MaritalStatus"
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Churn"):

    input_df = pd.DataFrame([{
        "Tenure": st.session_state["Tenure"],
        "CityTier": st.session_state["CityTier"],
        "WarehouseToHome": st.session_state["WarehouseToHome"],
        "HourSpendOnApp": st.session_state["HourSpendOnApp"],
        "NumberOfDeviceRegistered": st.session_state["NumberOfDeviceRegistered"],
        "SatisfactionScore": st.session_state["SatisfactionScore"],
        "NumberOfAddress": st.session_state["NumberOfAddress"],
        "Complain": st.session_state["Complain"],
        "OrderAmountHikeFromlastYear": st.session_state["OrderAmountHikeFromlastYear"],
        "CouponUsed": st.session_state["CouponUsed"],
        "OrderCount": st.session_state["OrderCount"],
        "DaySinceLastOrder": st.session_state["DaySinceLastOrder"],
        "CashbackAmount": st.session_state["CashbackAmount"],
        "PreferredLoginDevice": st.session_state["PreferredLoginDevice"],
        "PreferredPaymentMode": st.session_state["PreferredPaymentMode"],
        "Gender": st.session_state["Gender"],
        "PreferedOrderCat": st.session_state["PreferedOrderCat"],
        "MaritalStatus": st.session_state["MaritalStatus"],
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ Customer Likely to Churn\n\nChurn Risk Score: {probability*100:.1f}%%")
    else:
        st.success(f"✅ Customer Likely to Stay\n\nChurn Risk: {probability*100:.1f}%")