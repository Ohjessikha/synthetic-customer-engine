
import numpy as np
import pandas as pd


def find_column(df, candidates):
    """Find a column using common business-related names."""
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def get_generation_columns(df):
    """Identify columns used by customer generation controls."""
    return {
        "segment": find_column(
            df, ["customer_segment", "segment", "customer_type"]
        ),
        "spending": find_column(
            df, ["total_spending", "spending", "total_spend", "amount_spent"]
        ),
        "frequency": find_column(
            df, ["purchase_frequency", "purchase_count", "frequency"]
        ),
    }


def apply_generation_conditions(df, conditions):
    """Filter generated rows using the requested conditions."""
    result = df.copy()
    columns = get_generation_columns(result)

    segment_col = columns["segment"]
    if conditions.get("segment") and segment_col:
        result = result[
            result[segment_col].astype(str) == str(conditions["segment"])
        ]

    spending_col = columns["spending"]
    if spending_col:
        values = pd.to_numeric(result[spending_col], errors="coerce")
        minimum = conditions.get("min_spending")
        maximum = conditions.get("max_spending")

        if minimum is not None:
            result = result[values >= minimum]
        if maximum is not None:
            values = pd.to_numeric(result[spending_col], errors="coerce")
            result = result[values <= maximum]

    frequency_col = columns["frequency"]
    if frequency_col:
        values = pd.to_numeric(result[frequency_col], errors="coerce")
        minimum = conditions.get("min_frequency")
        maximum = conditions.get("max_frequency")

        if minimum is not None:
            result = result[values >= minimum]
        if maximum is not None:
            values = pd.to_numeric(result[frequency_col], errors="coerce")
            result = result[values <= maximum]

    return result


def statistical_baseline(df, number_of_rows, seed=42):
    """Resample rows and add small noise to numeric non-ID columns."""
    rng = np.random.default_rng(seed)
    result = df.sample(
        n=number_of_rows, replace=True, random_state=seed
    ).reset_index(drop=True)

    for column in result.select_dtypes(include="number").columns:
        name = str(column).strip().lower()
        if name == "id" or name.endswith("_id"):
            continue

        original = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(original) < 2 or original.nunique() < 2:
            continue

        std = original.std()
        if not np.isfinite(std) or std == 0:
            continue

        noise = rng.normal(0, std * 0.02, size=number_of_rows)
        generated = pd.to_numeric(result[column], errors="coerce") + noise

        if pd.api.types.is_integer_dtype(df[column].dtype):
            generated = generated.round().astype("Int64")

        result[column] = generated

    return result


def refresh_identifier_columns(generated, original):
    """Give generated customer IDs a synthetic prefix."""
    result = generated.copy()

    for column in original.columns:
        name = str(column).strip().lower()
        if name == "customer_id" and column in result.columns:
            result[column] = [
                f"SYNTH_{i:06d}" for i in range(1, len(result) + 1)
            ]

    return result


def generate_synthetic_customers(
    original_df, number_of_rows, conditions=None, seed=42
):
    """Generate synthetic customer rows and report the method used."""
    if original_df is None or original_df.empty:
        raise ValueError("The source dataset is empty.")

    if (
        not isinstance(number_of_rows, int)
        or isinstance(number_of_rows, bool)
    ):
        raise ValueError("Record count must be an integer.")

    if not 1 <= number_of_rows <= 5000:
        raise ValueError("Choose between 1 and 5,000 records.")

    conditions = conditions or {}

    for minimum_key, maximum_key in [
        ("min_spending", "max_spending"),
        ("min_frequency", "max_frequency"),
    ]:
        minimum = conditions.get(minimum_key)
        maximum = conditions.get(maximum_key)
        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"{minimum_key} cannot exceed {maximum_key}."
            )

    model_df = original_df.dropna(axis=1, how="all").copy()
    if model_df.empty or model_df.shape[1] == 0:
        raise ValueError("The source dataset contains no usable columns.")

    method = "Statistical baseline"
    explanation = (
        "Rows were resampled from the source data with small amounts "
        "of numerical noise. This is a demonstration baseline, not a "
        "trained generative model and it does not guarantee privacy."
    )

    generated = statistical_baseline(
        model_df, min(max(number_of_rows * 10, number_of_rows + 100), 25000),
        seed=seed,
    )

    for column in original_df.columns:
        if column not in generated.columns:
            generated[column] = np.nan
    generated = generated[original_df.columns]

    generated = apply_generation_conditions(generated, conditions)

    attempts = 0
    while len(generated) < number_of_rows and attempts < 4:
        attempts += 1
        extra = statistical_baseline(
            model_df,
            min(max(number_of_rows * 10, number_of_rows + 100), 25000),
            seed=seed + attempts,
        )
        for column in original_df.columns:
            if column not in extra.columns:
                extra[column] = np.nan
        extra = extra[original_df.columns]
        extra = apply_generation_conditions(extra, conditions)
        generated = pd.concat([generated, extra], ignore_index=True)

    generated = generated.head(number_of_rows).reset_index(drop=True)
    generated = refresh_identifier_columns(generated, original_df)

    if len(generated) < number_of_rows:
        explanation += (
            f" Only {len(generated)} of {number_of_rows} requested "
            "records satisfied the conditions. Try relaxing the filters."
        )

    return generated, method, explanation
