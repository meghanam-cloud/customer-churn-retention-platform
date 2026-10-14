import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn & Retention Intelligence")
st.caption("Predict customer churn using your trained CatBoost model.")

with st.form("customer_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=75.5, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=906.0, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn", type="primary", use_container_width=True
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges_clean": total_charges,
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=customer,
                timeout=30,
            )

        if response.ok:
            result = response.json()
            probability = result["churn_probability"]
            prediction = result["churn_prediction"]

            st.subheader("Prediction Results")

            metric1, metric2 = st.columns(2)
            metric1.metric("Predicted Churn", prediction)
            metric2.metric("Churn Probability", f"{probability:.1%}")

            st.progress(probability)

            if prediction == "Yes":
                st.warning(
                    "This customer is classified as at risk of churn. "
                    "Consider reviewing possible retention options."
                )
            else:
                st.success(
                    "This customer is not classified as at risk "
                    "at the current decision threshold."
                )

            st.caption(
                f"Model: {result['model_name']} | "
                f"Decision threshold: {result['decision_threshold']}"
            )
        else:
            st.error(
                f"API returned {response.status_code}: {response.text}"
            )

    except requests.RequestException as error:
        st.error(
            "Cannot connect to FastAPI. Start the API in another "
            f"terminal and try again. Details: {error}"
        )