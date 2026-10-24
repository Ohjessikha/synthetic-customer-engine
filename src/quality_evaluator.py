
import numpy as np
import pandas as pd


def compare_numeric_statistics(original_df, synthetic_df):
    """Compare summary statistics for shared numerical columns."""
    numeric_columns = original_df.select_dtypes(
        include="number"
    ).columns.intersection(
        synthetic_df.select_dtypes(include="number").columns
    )

    rows = []

    for column in numeric_columns:
        original = pd.to_numeric(
            original_df[column], errors="coerce"
        ).dropna()

        synthetic = pd.to_numeric(
            synthetic_df[column], errors="coerce"
        ).dropna()

        if original.empty or synthetic.empty:
            continue

        original_mean = original.mean()
        synthetic_mean = synthetic.mean()

        original_std = original.std()
        synthetic_std = synthetic.std()

        mean_difference_pct = (
            abs(synthetic_mean - original_mean)
            / max(abs(original_mean), 1e-9)
        ) * 100

        rows.append({
            "Column": column,
            "Original Mean": original_mean,
            "Synthetic Mean": synthetic_mean,
            "Original Median": original.median(),
            "Synthetic Median": synthetic.median(),
            "Original Std": original_std,
            "Synthetic Std": synthetic_std,
            "Mean Difference (%)": mean_difference_pct,
        })

    return pd.DataFrame(rows)


def compare_categorical_distributions(original_df, synthetic_df):
    """Compare category percentages for shared categorical columns."""
    categorical_columns = original_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.intersection(
        synthetic_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    comparisons = {}

    for column in categorical_columns:
        original_pct = (
            original_df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts(normalize=True)
            .mul(100)
        )

        synthetic_pct = (
            synthetic_df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts(normalize=True)
            .mul(100)
        )

        comparison = pd.concat(
            [
                original_pct.rename("Original (%)"),
                synthetic_pct.rename("Synthetic (%)"),
            ],
            axis=1,
        ).fillna(0)

        comparison.index.name = "Category"
        comparisons[column] = comparison.reset_index()

    return comparisons


def calculate_quality_summary(numeric_comparison):
    """Summarize numerical mean differences descriptively."""
    if numeric_comparison.empty:
        return {
            "columns_compared": 0,
            "average_mean_difference_pct": None,
        }

    return {
        "columns_compared": len(numeric_comparison),
        "average_mean_difference_pct": float(
            numeric_comparison["Mean Difference (%)"].mean()
        ),
    }
