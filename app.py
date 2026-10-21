
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import (
    generate_sample_data,
    get_dataset_summary,
    load_customer_csv,
)

from data_analysis import (
    cluster_customers,
    detect_anomalies,
    get_categorical_columns,
    get_categorical_summary,
    get_correlation_matrix,
    get_numeric_columns,
    get_numeric_summary,
    get_underrepresented_categories,
)


st.set_page_config(
    page_title="Synthetic Customer Engine",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Synthetic Customer Behavior & Data Augmentation Engine")

st.write(
    "Analyze customer behavior, discover customer groups, "
    "and identify unusual or underrepresented patterns."
)

st.info(
    "Milestone 4 adds customer behavior analysis. "
    "Synthetic data generation and GenAI will be added later."
)

st.header("1. Choose your dataset")

data_source = st.radio(
    "How would you like to provide customer data?",
    ["Use sample data", "Upload a CSV file"],
    horizontal=True,
)

df = None

if data_source == "Use sample data":
    number_of_rows = st.slider(
        "Number of sample customers",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
    )

    if st.button("Generate sample dataset"):
        st.session_state["sample_df"] = generate_sample_data(
            number_of_rows
        )

    if "sample_df" not in st.session_state:
        st.session_state["sample_df"] = generate_sample_data(200)

    df = st.session_state["sample_df"]

else:
    uploaded_file = st.file_uploader(
        "Upload your customer CSV",
        type=["csv"],
        help="Maximum file size: 10 MB.",
    )

    if uploaded_file is not None:
        try:
            df = load_customer_csv(uploaded_file)
            st.session_state["uploaded_df"] = df
            st.success("CSV uploaded successfully.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "Could not process this file. "
                "Please check its format and try again."
            )

    elif "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]

if df is not None:
    st.divider()

    st.header("2. Dataset overview")

    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])
    col3.metric("Missing cells", summary["missing_cells"])
    col4.metric("Duplicate rows", summary["duplicate_rows"])

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": [str(dtype) for dtype in df.dtypes],
            "Missing values": [
                int(count) for count in df.isna().sum()
            ],
            "Unique values": [
                int(df[column].nunique(dropna=True))
                for column in df.columns
            ],
        }
    )

    with st.expander("View column information"):
        st.dataframe(column_info, use_container_width=True)

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)

    st.divider()
    st.header("3. Customer behavior analysis")

    if numeric_columns:
        st.subheader("Numerical feature statistics")
        st.dataframe(
            get_numeric_summary(df).round(2),
            use_container_width=True,
        )

        selected_chart = st.selectbox(
            "Choose a numerical feature to explore",
            numeric_columns,
        )

        chart_data = pd.to_numeric(
            df[selected_chart], errors="coerce"
        ).dropna()

        if not chart_data.empty:
            fig = px.histogram(
                chart_data,
                x=selected_chart,
                nbins=25,
                title=f"Distribution of {selected_chart}",
                labels={selected_chart: selected_chart},
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "No suitable numerical features were found. "
            "Upload a dataset with numeric columns to view these analyses."
        )

    if categorical_columns:
        st.subheader("Categorical feature analysis")

        categorical_summary = get_categorical_summary(df)

        st.dataframe(
            categorical_summary,
            use_container_width=True,
        )

        selected_category = st.selectbox(
            "Choose a categorical feature to visualize",
            categorical_columns,
            key="category_chart",
        )

        category_counts = (
            df[selected_category]
            .fillna("(Missing)")
            .astype(str)
            .value_counts()
            .head(15)
            .rename_axis("Category")
            .reset_index(name="Count")
        )

        fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            title=f"Customer counts by {selected_category}",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation analysis")

    correlation = get_correlation_matrix(df)

    if not correlation.empty:
        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numerical feature correlation matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Correlation measures linear association, not causation."
        )
    else:
        st.info(
            "At least two suitable numerical features are needed "
            "for correlation analysis."
        )

    st.divider()
    st.header("4. Customer segmentation")

    if len(numeric_columns) >= 2 and len(df) >= 3:
        default_features = [
            column
            for column in [
                "age",
                "purchase_frequency",
                "total_spending",
            ]
            if column in numeric_columns
        ]

        if len(default_features) < 2:
            default_features = numeric_columns[:min(3, len(numeric_columns))]

        cluster_features = st.multiselect(
            "Select features for clustering",
            options=numeric_columns,
            default=default_features,
            key="cluster_features",
        )

        number_of_clusters = st.slider(
            "Number of customer clusters",
            min_value=2,
            max_value=min(10, len(df) - 1),
            value=min(3, len(df) - 1),
        )

        if st.button("Analyze customer segments"):
            try:
                clustered_df = cluster_customers(
                    df,
                    cluster_features,
                    number_of_clusters,
                )

                st.session_state["clustered_df"] = clustered_df
                st.session_state["cluster_features_used"] = (
                    cluster_features
                )

            except ValueError as exc:
                st.error(str(exc))

        if "clustered_df" in st.session_state:
            clustered_df = st.session_state["clustered_df"]

            if (
                "Customer Cluster" in clustered_df.columns
                and len(clustered_df) == len(df)
            ):
                st.subheader("Cluster distribution")

                cluster_counts = (
                    clustered_df["Customer Cluster"]
                    .value_counts()
                    .rename_axis("Cluster")
                    .reset_index(name="Customers")
                )

                st.dataframe(
                    cluster_counts,
                    use_container_width=True,
                )

                if len(cluster_features) >= 2:
                    fig = px.scatter(
                        clustered_df,
                        x=cluster_features[0],
                        y=cluster_features[1],
                        color="Customer Cluster",
                        hover_data=cluster_features,
                        title="Customer clusters",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                st.dataframe(
                    clustered_df.head(50),
                    use_container_width=True,
                )

                st.caption(
                    "Cluster labels are algorithm-generated groups, "
                    "not verified customer personas."
                )
    else:
        st.info(
            "Clustering requires at least two suitable numeric "
            "features and three records."
        )

    st.divider()
    st.header("5. Underrepresented customer groups")

    minimum_percentage = st.slider(
        "Show categories occurring below this percentage",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    rare_categories = get_underrepresented_categories(
        df,
        minimum_percentage,
    )

    if rare_categories.empty:
        st.success(
            "No categorical values fall below the selected threshold."
        )
    else:
        st.dataframe(
            rare_categories,
            use_container_width=True,
        )

    st.divider()
    st.header("6. Potentially unusual records")

    if len(numeric_columns) >= 2 and len(df) >= 10:
        anomaly_features = st.multiselect(
            "Choose numerical features for anomaly detection",
            options=numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            key="anomaly_features",
        )

        contamination_percent = st.slider(
            "Expected proportion of unusual records (%)",
            min_value=1,
            max_value=20,
            value=5,
        )

        if st.button("Detect unusual records"):
            try:
                anomaly_df = detect_anomalies(
                    df,
                    anomaly_features,
                    contamination=contamination_percent / 100,
                )

                st.session_state["anomaly_df"] = anomaly_df

            except ValueError as exc:
                st.error(str(exc))

        if "anomaly_df" in st.session_state:
            anomaly_df = st.session_state["anomaly_df"]

            if (
                "Anomaly Status" in anomaly_df.columns
                and len(anomaly_df) == len(df)
            ):
                unusual_count = int(
                    (
                        anomaly_df["Anomaly Status"]
                        == "Potentially unusual"
                    ).sum()
                )

                st.metric(
                    "Potentially unusual records",
                    unusual_count,
                )

                st.dataframe(
                    anomaly_df[
                        anomaly_df["Anomaly Status"]
                        == "Potentially unusual"
                    ].head(50),
                    use_container_width=True,
                )

                st.caption(
                    "These are statistical flags, not proof of errors "
                    "or fraudulent activity."
                )
    else:
        st.info(
            "Anomaly detection requires at least two suitable numeric "
            "features and ten records."
        )

    st.divider()
    st.header("7. Download dataset")

    st.download_button(
        "Download current dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="customer_dataset.csv",
        mime="text/csv",
    )

st.divider()

st.caption(
    "Milestone 4: Customer behavior analysis, segmentation, "
    "correlations, rare categories, and anomaly detection."
)
