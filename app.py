
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import (
    generate_sample_data,
    get_dataset_summary,
    load_customer_csv,
)


st.set_page_config(
    page_title="Synthetic Customer Engine",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Synthetic Customer Behavior & Data Augmentation Engine")

st.write(
    "Explore customer data, identify data-quality issues, "
    "and prepare datasets for synthetic data generation."
)

st.info(
    "This milestone implements dataset loading, basic validation, "
    "and fictional sample-data generation."
)

st.header("1. Choose your dataset")

data_source = st.radio(
    "How would you like to provide customer data?",
    ["Use sample data", "Upload a CSV file"],
    horizontal=True,
)

df = st.session_state["sample_df"]

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
        st.session_state["data_source"] = "Sample data"

    if "sample_df" not in st.session_state:
        st.session_state["sample_df"] = generate_sample_data(200)
        st.session_state["data_source"] = "Sample data"

    df = st.session_state["sample_df"]

else:
    uploaded_file = st.file_uploader(
        "Upload your customer dataset",
        type=["csv"],
        help="Maximum file size: 10 MB. Use fictional or authorized data.",
    )

    if uploaded_file is not None:
        try:
            df = load_customer_csv(uploaded_file)
            st.session_state["customer_df"] = df
            st.session_state["data_source"] = "Uploaded CSV"
            st.success("CSV uploaded and basic validation passed.")

        except ValueError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(
                "An unexpected error occurred while reading the file. "
                "Check the CSV format and try again."
            )

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

    st.subheader("Column information")

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": [
                str(dtype) for dtype in df.dtypes
            ],
            "Missing values": [
                int(count)
                for count in df.isna().sum()
            ],
            "Unique values": [
                int(df[column].nunique(dropna=True))
                for column in df.columns
            ],
        }
    )

    st.dataframe(column_info, use_container_width=True)

    st.subheader("Data-quality checks")

    if summary["missing_cells"] == 0:
        st.success("No missing cells detected.")
    else:
        st.warning(
            f"Found {summary['missing_cells']} missing cells. "
            "They will need further inspection."
        )

    if summary["duplicate_rows"] == 0:
        st.success("No completely duplicated rows detected.")
    else:
        st.warning(
            f"Found {summary['duplicate_rows']} completely duplicated rows."
        )

    with st.expander("View missing values by column"):
        missing_df = pd.DataFrame(
            {
                "Column": summary["missing_by_column"].keys(),
                "Missing values": summary["missing_by_column"].values(),
            }
        )
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Download dataset")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download current dataset as CSV",
        data=csv_data,
        file_name="customer_dataset.csv",
        mime="text/csv",
    )

    st.caption(
        "This download contains the currently selected dataset. "
        "It does not contain generated synthetic records from a separate engine."
    )

st.divider()

st.caption(
    "Milestone 3: Dataset upload, basic data-quality checks, "
    "and sample-data generation."
)
