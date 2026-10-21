
from io import BytesIO

import numpy as np
import pandas as pd


MAX_FILE_SIZE_MB = 10


def generate_sample_data(number_of_rows=200, seed=42):
    """Generate fictional customer records for demonstration."""

    if not isinstance(number_of_rows, int) or isinstance(
        number_of_rows, bool
    ):
        raise ValueError("The number of rows must be an integer.")

    if not 1 <= number_of_rows <= 10000:
        raise ValueError("Choose between 1 and 10,000 rows.")

    rng = np.random.default_rng(seed)

    ages = rng.integers(18, 71, size=number_of_rows)
    frequency = rng.poisson(lam=5, size=number_of_rows)
    frequency = np.clip(frequency, 0, 30)

    spending = (
        frequency * rng.uniform(250, 1500, size=number_of_rows)
    ).round(2)

    segments = np.where(
        (frequency >= 8) & (spending >= 7000),
        "High Value",
        np.where(frequency <= 2, "Occasional", "Regular"),
    )

    return pd.DataFrame(
        {
            "customer_id": [
                f"CUST{i:04d}"
                for i in range(1, number_of_rows + 1)
            ],
            "age": ages,
            "purchase_frequency": frequency,
            "total_spending": spending,
            "customer_segment": segments,
        }
    )


def load_customer_csv(uploaded_file):
    """Read and validate an uploaded customer CSV file."""

    if uploaded_file is None:
        raise ValueError("Please upload a CSV file.")

    filename = getattr(uploaded_file, "name", "")
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError("The file must be 10 MB or smaller.")

    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "The CSV does not contain readable column names or data."
        ) from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            "The file could not be parsed. Check its CSV format and encoding."
        ) from exc

    if df.empty:
        raise ValueError(
            "The CSV contains no data rows. Please upload a non-empty dataset."
        )

    if len(df.columns) == 0:
        raise ValueError("The CSV must contain at least one column.")

    original_columns = list(df.columns)
    cleaned_columns = [
        str(column).strip() for column in original_columns
    ]

    if any(not column for column in cleaned_columns):
        raise ValueError("Column names cannot be blank.")

    if len(set(cleaned_columns)) != len(cleaned_columns):
        raise ValueError(
            "Column names must be unique after removing surrounding spaces."
        )

    df.columns = cleaned_columns

    return df


def get_dataset_summary(df):
    """Return basic data-quality information."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": list(df.columns),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
    }
