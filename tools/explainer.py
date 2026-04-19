"""
Tool 2: explain_prediction
Uses SHAP TreeExplainer to compute per-prediction feature contributions.
Returns ranked SHAP values so the LLM can reason about WHY the customer churns.
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
from langchain_core.tools import tool

from src.config import PIPELINE_PATH

warnings.filterwarnings("ignore")

DIRECTION_MAP = {True: "pushes_toward_churn", False: "pushes_away_from_churn"}


# Global cache for the pipeline
_pipeline = None


@tool
def explain_prediction(customer_data: str) -> str:
    """Explain the churn prediction using SHAP (SHapley Additive exPlanations).

    Computes per-feature SHAP values for this specific customer, showing
    exactly which features push the prediction toward or away from churn.

    Args:
        customer_data: A JSON string containing customer attributes.

    Returns:
        JSON string with a list of features ranked by SHAP impact,
        each containing: feature name, customer's raw value,
        SHAP value, absolute impact, and direction.
    """
    global _pipeline

    if _pipeline is None:
        try:
            _pipeline = joblib.load(PIPELINE_PATH)
        except Exception as e:
            return json.dumps({"error": f"Failed to load pipeline for explanation: {str(e)}"})

    pipeline = _pipeline
    data = json.loads(customer_data)

    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    df = pd.DataFrame([data])

    # Transform the data using the pipeline's preprocessor
    X_transformed = preprocessor.transform(df)

    # Convert sparse matrix to dense array for SHAP compatibility
    to_dense = getattr(X_transformed, "toarray", lambda: np.array(X_transformed))
    X_array = to_dense()

    # Get feature names after preprocessing (remove prefixes like 'num__')
    raw_feature_names = list(preprocessor.get_feature_names_out())
    clean_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in raw_feature_names
    ]

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_array)

    # Handle both single-array and list-of-arrays output formats
    shap_row = np.array(shap_values)
    while shap_row.ndim > 1:
        shap_row = shap_row[0]

    # Build contributions list — dict lookup for direction, no if-else
    contributions = []
    for i, (feature_name, shap_val) in enumerate(zip(clean_names, shap_row)):
        # Get the customer's raw value for this feature from input data
        raw_value = data.get(feature_name, round(float(X_array[0][i]), 4))
        contributions.append({
            "feature": feature_name,
            "customer_value": raw_value,
            "shap_value": round(float(shap_val), 4),
            "absolute_impact": round(abs(float(shap_val)), 4),
            "direction": DIRECTION_MAP[float(shap_val) > 0],
        })

    # Sort by absolute SHAP value (highest impact first)
    contributions.sort(key=lambda x: x["absolute_impact"], reverse=True)

    # Include the base value for context
    result = {
        "base_value": round(float(explainer.expected_value), 4),
        "note": "base_value is the average model output; SHAP values show deviation from it",
        "feature_contributions": contributions,
    }

    return json.dumps(result, indent=2)