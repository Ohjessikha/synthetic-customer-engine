
import pandas as pd


def find_column(df, candidates):
    """Find a column using common business-related names."""
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    return None


def generate_customer_insights(df):
    """Create data-grounded customer insights using simple rules."""

    if df is None or df.empty:
        return [
            {
                "title": "No data available",
                "insight": "Load a non-empty dataset to generate insights.",
                "recommendation": "Upload a valid customer dataset or select sample data.",
            }
        ]

    insights = []

    spending_col = find_column(
        df,
        ["total_spending", "spending", "total_spend", "amount_spent"],
    )

    frequency_col = find_column(
        df,
        ["purchase_frequency", "purchase_count", "frequency"],
    )

    segment_col = find_column(
        df,
        ["customer_segment", "segment", "customer_type"],
    )

    # Insight 1: Spending analysis
    if spending_col:
        spending = pd.to_numeric(
            df[spending_col], errors="coerce"
        ).dropna()

        if not spending.empty:
            average_spending = spending.mean()
            median_spending = spending.median()

            insights.append({
                "title": "Customer spending",
                "insight": (
                    f"Average spending is {average_spending:,.2f}, "
                    f"while median spending is {median_spending:,.2f}."
                ),
                "recommendation": (
                    "Compare the average and median to understand "
                    "whether a small number of high-spending customers "
                    "may be influencing the average."
                ),
            })

    # Insight 2: Purchase frequency
    if frequency_col:
        frequency = pd.to_numeric(
            df[frequency_col], errors="coerce"
        ).dropna()

        if not frequency.empty:
            average_frequency = frequency.mean()

            insights.append({
                "title": "Purchase frequency",
                "insight": (
                    f"Average purchase frequency is "
                    f"{average_frequency:,.2f}."
                ),
                "recommendation": (
                    "Review purchase frequency across customer groups "
                    "before deciding where retention campaigns may help."
                ),
            })

    # Insight 3: Customer segments
    if segment_col:
        segments = df[segment_col].dropna().astype(str)

        if not segments.empty:
            counts = segments.value_counts()
            largest_segment = counts.index[0]
            percentage = counts.iloc[0] / len(segments) * 100

            insights.append({
                "title": "Largest customer segment",
                "insight": (
                    f"{largest_segment} is the largest segment, "
                    f"representing {percentage:.1f}% of customers "
                    f"with a recorded segment."
                ),
                "recommendation": (
                    "Consider whether this segment needs a dedicated "
                    "engagement strategy, while also checking the needs "
                    "of smaller segments."
                ),
            })

    # Insight 4: Overall dataset
    insights.append({
        "title": "Dataset overview",
        "insight": (
            f"The dataset contains {len(df):,} rows and "
            f"{len(df.columns):,} columns."
        ),
        "recommendation": (
            "Check missing values and data quality before using "
            "these observations to make business decisions."
        ),
    })

    if not insights:
        insights.append({
            "title": "Limited insights",
            "insight": (
                "The dataset does not contain recognizable spending, "
                "frequency, or customer-segment columns."
            ),
            "recommendation": (
                "Use column names such as total_spending, "
                "purchase_frequency, or customer_segment where appropriate."
            ),
        })

    return insights
