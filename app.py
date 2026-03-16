import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and expected feature schema
model = joblib.load("final_churn_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("Telco Customer Churn Prediction App")
st.write(
    "Provide the key customer details below to predict churn risk, understand likely drivers, and view recommended retention actions."
)

st.info(
    "Complete the Core Risk Inputs first. Advanced Inputs are optional, but can improve the realism of the prediction."
)

# Helper functions
PLACEHOLDER = "........."


def create_tenure_group(tenure):
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 48:
        return "25-48"
    return "49-72"


def create_monthly_band(monthly_charge):
    if monthly_charge <= 35:
        return "Low"
    elif monthly_charge <= 65:
        return "Mid-Low"
    elif monthly_charge <= 90:
        return "Mid-High"
    return "High"


def parse_non_negative_number(value, field_name, is_int=False):
    value = value.strip()
    if value == "":
        raise ValueError(f"{field_name} is required.")
    try:
        number = int(value) if is_int else float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid number.")
    if number < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return number


def fill_optional_select(value, default):
    return default if value == PLACEHOLDER else value


def fill_optional_numeric(value, default, is_int=False):
    value = value.strip()
    if value == "":
        return default
    try:
        return int(value) if is_int else float(value)
    except ValueError:
        return default


def explain_churn_risk(input_data):
    risk_factors = []
    protective_factors = []

    # Primary risk drivers
    if input_data["Contract"] == "Month-to-month":
        risk_factors.append("Month-to-month contract indicates lower customer commitment.")
    if input_data["tenure_group"] == "0-12":
        risk_factors.append("Very short tenure places the customer in the highest-risk churn period.")
    if input_data["MonthlyCharges_band"] == "High":
        risk_factors.append("High monthly charges may increase price sensitivity.")
    if input_data["Payment Method"] == "Electronic check":
        risk_factors.append("Electronic check payments are historically associated with higher churn.")
    if input_data["Internet Service"] == "Fiber optic":
        risk_factors.append("Fiber optic customers often show higher churn due to price or service expectations.")
    if input_data["Online Security"] == "No":
        risk_factors.append("No online security suggests weaker service attachment.")
    if input_data["Tech Support"] == "No":
        risk_factors.append("No tech support may indicate lower engagement with value-added services.")

    # Secondary context
    if input_data["Device Protection"] == "No":
        risk_factors.append("No device protection reduces service stickiness.")
    if input_data["Partner"] == "No":
        risk_factors.append("Customers without partners may face fewer switching barriers.")
    if input_data["Dependents"] == "No":
        risk_factors.append("Customers without dependents may be more flexible to switch providers.")

    # Protective signals
    if input_data["Contract"] in ["One year", "Two year"]:
        protective_factors.append("Long-term contract reduces churn likelihood.")
    if input_data["tenure_group"] in ["25-48", "49-72"]:
        protective_factors.append("Long tenure suggests customer loyalty.")
    if input_data["Online Security"] == "Yes":
        protective_factors.append("Online security increases product stickiness.")
    if input_data["Tech Support"] == "Yes":
        protective_factors.append("Tech support usage indicates deeper service engagement.")
    if input_data["Device Protection"] == "Yes":
        protective_factors.append("Device protection adds dependency on the service.")

    return risk_factors[:5], protective_factors[:3]


def generate_recommendations(input_data, probability):
    recommendations = []

    if probability >= 0.75:
        recommendations.append("Prioritize this customer for immediate retention outreach.")
    elif probability >= 0.40:
        recommendations.append("Monitor this customer closely and target them with personalized retention offers.")
    else:
        recommendations.append("Maintain routine engagement and continue monitoring for changes in churn risk.")

    if input_data["Contract"] == "Month-to-month":
        recommendations.append("Offer an incentive to move the customer to a one-year or two-year contract.")
    if input_data["MonthlyCharges_band"] == "High":
        recommendations.append("Review pricing, discounts, or bundle offers to reduce perceived cost burden.")
    if input_data["Online Security"] == "No" or input_data["Tech Support"] == "No":
        recommendations.append("Promote bundles that include online security or tech support to increase stickiness.")
    if input_data["Payment Method"] == "Electronic check":
        recommendations.append("Encourage a switch to an automatic payment method to improve retention.")
    if input_data["tenure_group"] == "0-12":
        recommendations.append("Provide onboarding support or early-life-cycle loyalty incentives.")
    if input_data["Internet Service"] == "Fiber optic":
        recommendations.append("Check for service quality or pricing concerns among fiber customers.")

    return recommendations


# Core Risk Inputs (Mandatory)
st.subheader("Core Risk Inputs")

core_col1, core_col2 = st.columns(2)

with core_col1:
    tenure_months_text = st.text_input("Tenure Months *", placeholder="Enter tenure in months")
    monthly_charges_text = st.text_input("Monthly Charges *", placeholder="Enter monthly charges")
    contract = st.selectbox(
        "Contract *",
        [PLACEHOLDER, "Month-to-month", "One year", "Two year"],
        index=0,
    )
    payment_method = st.selectbox(
        "Payment Method *",
        [PLACEHOLDER, "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=0,
    )

with core_col2:
    internet_service = st.selectbox(
        "Internet Service *",
        [PLACEHOLDER, "DSL", "Fiber optic", "No"],
        index=0,
    )
    online_security = st.selectbox(
        "Online Security *",
        [PLACEHOLDER, "Yes", "No", "No internet service"],
        index=0,
    )
    tech_support = st.selectbox(
        "Tech Support *",
        [PLACEHOLDER, "Yes", "No", "No internet service"],
        index=0,
    )

# Advanced Inputs (Optional)
with st.expander("Advanced Customer Inputs (Optional)"):
    adv_col1, adv_col2, adv_col3 = st.columns(3)

    with adv_col1:
        gender = st.selectbox("Gender", [PLACEHOLDER, "Male", "Female"], index=0)
        senior_citizen = st.selectbox("Senior Citizen", [PLACEHOLDER, "Yes", "No"], index=0)
        partner = st.selectbox("Partner", [PLACEHOLDER, "Yes", "No"], index=0)
        dependents = st.selectbox("Dependents", [PLACEHOLDER, "Yes", "No"], index=0)

    with adv_col2:
        phone_service = st.selectbox("Phone Service", [PLACEHOLDER, "Yes", "No"], index=0)
        multiple_lines = st.selectbox(
            "Multiple Lines",
            [PLACEHOLDER, "No", "Yes", "No phone service"],
            index=0,
        )
        online_backup = st.selectbox(
            "Online Backup",
            [PLACEHOLDER, "No", "Yes", "No internet service"],
            index=0,
        )
        device_protection = st.selectbox(
            "Device Protection",
            [PLACEHOLDER, "No", "Yes", "No internet service"],
            index=0,
        )

    with adv_col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            [PLACEHOLDER, "No", "Yes", "No internet service"],
            index=0,
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            [PLACEHOLDER, "No", "Yes", "No internet service"],
            index=0,
        )
        paperless_billing = st.selectbox(
            "Paperless Billing",
            [PLACEHOLDER, "Yes", "No"],
            index=0,
        )
        total_charges_text = st.text_input("Total Charges", placeholder="Optional")
        country = st.text_input("Country", placeholder="Optional")
        state = st.text_input("State", placeholder="Optional")
        city = st.text_input("City", placeholder="Optional")

# Predict
if st.button("Predict Churn"):
    # Validate required fields
    required_selects = [contract, payment_method, internet_service, online_security, tech_support]
    if PLACEHOLDER in required_selects:
        st.warning("Please complete all required dropdown fields before prediction.")
        st.stop()

    try:
        tenure_months = parse_non_negative_number(tenure_months_text, "Tenure Months", is_int=True)
        monthly_charges = parse_non_negative_number(monthly_charges_text, "Monthly Charges", is_int=False)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    # Optional values fallback defaults
    total_charges = fill_optional_numeric(total_charges_text, default=0.0, is_int=False)

    input_data = {
        "Count": 1,
        "Country": country.strip() if country.strip() else "United States",
        "State": state.strip() if state.strip() else "California",
        "City": city.strip() if city.strip() else "Los Angeles",
        "Gender": fill_optional_select(gender, "Female"),
        "Senior Citizen": fill_optional_select(senior_citizen, "No"),
        "Partner": fill_optional_select(partner, "No"),
        "Dependents": fill_optional_select(dependents, "No"),
        "Tenure Months": tenure_months,
        "Phone Service": fill_optional_select(phone_service, "Yes"),
        "Multiple Lines": fill_optional_select(multiple_lines, "No"),
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": fill_optional_select(online_backup, "No"),
        "Device Protection": fill_optional_select(device_protection, "No"),
        "Tech Support": tech_support,
        "Streaming TV": fill_optional_select(streaming_tv, "No"),
        "Streaming Movies": fill_optional_select(streaming_movies, "No"),
        "Contract": contract,
        "Paperless Billing": fill_optional_select(paperless_billing, "Yes"),
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "tenure_group": create_tenure_group(tenure_months),
        "MonthlyCharges_band": create_monthly_band(monthly_charges),
    }

    input_df = pd.DataFrame([input_data])

    # Ensure all expected columns exist
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # Match training feature order
    input_df = input_df[model_features]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Explanations and recommendations
    risk_factors, protective_factors = explain_churn_risk(input_data)
    recommendations = generate_recommendations(input_data, probability)

    # Results Dashboard
    st.divider()
    st.subheader("Prediction Dashboard")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with metric_col2:
        st.metric("Prediction", "Likely to Churn" if prediction == 1 else "Likely to Stay")

    with metric_col3:
        if probability >= 0.75:
            risk_label = "High"
        elif probability >= 0.40:
            risk_label = "Medium"
        else:
            risk_label = "Low"
        st.metric("Risk Level", risk_label)

    st.subheader("Customer Churn Risk")
    st.progress(float(probability))

    if prediction == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")

    st.divider()
    st.subheader("Why this prediction was made")

    if risk_factors:
        st.markdown("### Main risk drivers")
        for factor in risk_factors:
            st.markdown(f"- {factor}")

    if protective_factors:
        st.markdown("### Main retention signals")
        for factor in protective_factors:
            st.markdown(f"- {factor}")

    st.divider()
    st.subheader("Recommended Actions")
    for rec in recommendations:
        st.markdown(f"✅ {rec}")

    with st.expander("Show submitted input data"):
        st.dataframe(input_df)