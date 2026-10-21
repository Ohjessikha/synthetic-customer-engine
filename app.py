
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Synthetic Customer Engine",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Synthetic Customer Behavior & Data Augmentation Engine")

st.write(
    "Analyze customer behavior, generate synthetic customer data, "
    "and evaluate the quality of artificial records."
)

st.info(
    "Welcome! This is the initial version of the application. "
    "Dataset analysis, synthetic data generation, and GenAI "
    "features will be added in upcoming milestones."
)

st.header("Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Application Status", "Running")

with col2:
    st.metric("Data Source", "Sample")

with col3:
    st.metric("Generation Engine", "Coming soon")

st.header("Sample Customer Dataset")

sample_data = pd.DataFrame(
    {
        "customer_id": [101, 102, 103, 104, 105],
        "age": [22, 35, 28, 42, 31],
        "purchase_frequency": [3, 8, 5, 12, 4],
        "total_spending": [1500, 7200, 3200, 10500, 2400],
        "customer_segment": [
            "Regular",
            "High Value",
            "Regular",
            "High Value",
            "Occasional",
        ],
    }
)

st.dataframe(sample_data, use_container_width=True)

st.caption(
    "This is fictional demonstration data, not real customer information."
)

st.success(
    "Milestone 2: Initial application interface is ready for testing."
)
