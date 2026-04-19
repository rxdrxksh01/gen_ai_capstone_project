import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Configuration ---
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.5

# --- File Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_PATH = str(PROJECT_ROOT / "models" / "churn_pipeline.pkl")
DATASET_PATH = str(PROJECT_ROOT / "data" / "E Commerce Dataset.xlsx")

# --- Feature Lists ---
NUMERICAL_FEATURES = [
    "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
    "OrderCount", "DaySinceLastOrder", "CashbackAmount"
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice", "PreferredPaymentMode",
    "Gender", "PreferedOrderCat", "MaritalStatus"
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
