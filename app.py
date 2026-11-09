import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_transactions
from src.features import build_customer_features
from src.predict import predict_customer


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Customer Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 16px;
        border-radius: 12px;
    }

    .segment-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">📊 Customer Clustering Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer segmentation using K-Means clustering'
    '</div>',
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Customer Prediction", "About the Project"]
)

st.sidebar.divider()

st.sidebar.info(
    "Upload transaction data for exactly one customer "
    "to predict their customer segment."
)


# ---------------- CUSTOMER PREDICTION PAGE ----------------

if page == "Customer Prediction":

    st.header("Predict Customer Segment")

    st.write(
        "Upload a CSV, Excel, or Parquet file containing the "
        "transactions of a single customer."
    )

    uploaded_file = st.file_uploader(
        "Upload customer transaction data",
        type=["csv", "xlsx", "parquet"]
    )

    if uploaded_file is not None:

        try:
            # Read the uploaded file
            transactions = pd.read_csv(uploaded_file) \
                if uploaded_file.name.lower().endswith(".csv") \
                else pd.read_excel(uploaded_file) \
                if uploaded_file.name.lower().endswith(".xlsx") \
                else pd.read_parquet(uploaded_file)

            st.success("File uploaded successfully!")

            with st.expander("Preview uploaded transactions"):
                st.dataframe(
                    transactions.head(10),
                    use_container_width=True
                )

            if st.button("Predict Customer Segment", type="primary"):

                with st.spinner("Analyzing customer transactions..."):

                    # Validate uploaded data
                    import tempfile
                    from pathlib import Path

                    suffix = Path(uploaded_file.name).suffix.lower()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    try:
                        validated_transactions = load_transactions(temp_path)
                    finally:
                        Path(temp_path).unlink(missing_ok=True)

                    # Generate customer-level features
                    customer_features = build_customer_features(
                        validated_transactions
                    )

                    # Predict customer segment
                    result = predict_customer(customer_features)

                st.divider()
                st.subheader("Prediction Result")

                # Main segment card
                st.markdown(
                    f"""
                    <div class="segment-card">
                        <h3>Predicted Segment</h3>
                        <h2>{result['segment_name']}</h2>
                        <p>Cluster ID: {result['cluster_id']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Customer metrics
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Days Since Last Purchase",
                    int(customer_features.iloc[0]["Recency"])
                )

                col2.metric(
                    "Purchase Frequency",
                    int(customer_features.iloc[0]["Sale_Frequency"])
                )

                col3.metric(
                    "Average Monthly Spending",
                    f"{customer_features.iloc[0]['Monthly_Spending_Mean']:.2f}"
                )

                col4, col5 = st.columns(2)

                col4.metric(
                    "Average Discount Rate",
                    f"{customer_features.iloc[0]['average_discount_rate']:.2%}"
                )

                col5.metric(
                    "Average Unit Price",
                    f"{customer_features.iloc[0]['Average_Unit_Price']:.2f}"
                )

                # Distance information
                st.divider()
                st.subheader("Model Confidence Details")

                st.write(
                    "The following distances show how far the customer "
                    "is from the cluster centers in the model's scaled "
                    "feature space. A smaller distance means a closer match."
                )

                st.metric(
                    "Distance to Assigned Cluster Center",
                    f"{result['distance_to_cluster_center']:.4f}"
                )

                st.write(
                    f"**Next closest cluster:** "
                    f"{result['runner_up_cluster']}  \n"
                    f"**Distance to next closest cluster:** "
                    f"{result['runner_up_distance']:.4f}"
                )

                # Cluster distances
                st.subheader("Distance from Each Cluster")

                distance_df = pd.DataFrame(
                    list(result["all_cluster_distances"].items()),
                    columns=["Cluster ID", "Distance"]
                )

                st.bar_chart(
                    distance_df.set_index("Cluster ID")
                )

                # Feature comparison
                st.subheader("Feature-Level Explanation")

                reasons = result.get("feature_reasons", [])

                if reasons:
                    reasons_df = pd.DataFrame(reasons)

                    st.write(
                        "These values compare the customer's scaled "
                        "features with the assigned cluster center."
                    )

                    st.dataframe(
                        reasons_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No feature-level explanation is available.")

                # Customer profile comparison
                st.subheader("Customer Profile Insights")

                profile_reasons = result.get("profile_reasons", [])

                if profile_reasons:
                    for reason in profile_reasons:
                        st.write(f"• {reason}")
                else:
                    st.info(
                        "No additional profile comparisons are available."
                    )

                # Download features
                st.divider()

                csv_data = customer_features.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Extracted Customer Features",
                    data=csv_data,
                    file_name="customer_features.csv",
                    mime="text/csv"
                )

        except Exception as exc:
            st.error(f"Could not process the uploaded file: {exc}")
            st.info(
                "Check that the file contains the required columns "
                "and transactions for exactly one customer."
            )


# ---------------- ABOUT PAGE ----------------

elif page == "About the Project":

    st.header("About This Project")

    st.write("""
    This application uses **K-Means clustering** to assign a customer
    to one of four customer segments based on their transaction behavior.

    ### Model input features

    - Recency
    - Sale Frequency
    - Monthly Spending Mean
    - Average Discount Rate
    - Average Unit Price

    ### Customer segments

    - **Loyal High-Value**
    - **Inactive Moderate-Spending Customers**
    - **Discount-Driven Regulars**
    - **Lapsed Low-Spending Customers**

    ### Workflow

    1. Upload transaction data for one customer.
    2. Validate the transaction data.
    3. Extract customer-level features.
    4. Apply the saved scaling parameters.
    5. Predict the customer cluster using the trained K-Means model.
    6. Display the assigned segment and supporting metrics.

    **Note:** The quality of predictions depends on using the same
    feature-engineering definitions and analysis date as the original
    model training pipeline.
    """)

