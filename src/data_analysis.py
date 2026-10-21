
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ID_WORDS = {"id", "customer_id", "user_id", "record_id"}


def get_numeric_columns(df):
    """Find numeric columns suitable for statistical analysis."""

    columns = []

    for column in df.select_dtypes(include="number").columns:
        normalized = str(column).strip().lower()

        if normalized in ID_WORDS or normalized.endswith("_id"):
            continue

        if df[column].notna().sum() >= 2:
            columns.append(column)

    return columns


def get_categorical_columns(df):
    """Find columns suitable for categorical analysis."""

    return [
        column
        for column in df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
        if df[column].notna().any()
    ]


def get_numeric_summary(df):
    """Calculate descriptive statistics for numeric columns."""

    columns = get_numeric_columns(df)

    if not columns:
        return pd.DataFrame()

    return df[columns].describe().T.reset_index(
        names="Feature"
    )


def get_categorical_summary(df, max_categories=10):
    """Count the most common values in categorical columns."""

    results = []

    for column in get_categorical_columns(df):
        counts = (
            df[column]
            .fillna("(Missing)")
            .astype(str)
            .value_counts()
            .head(max_categories)
        )

        for category, count in counts.items():
            results.append(
                {
                    "Feature": column,
                    "Category": category,
                    "Count": int(count),
                    "Percentage": round(
                        count / len(df) * 100, 2
                    ),
                }
            )

    return pd.DataFrame(results)


def get_correlation_matrix(df):
    """Calculate correlations between numeric features."""

    columns = get_numeric_columns(df)

    if len(columns) < 2:
        return pd.DataFrame()

    return df[columns].corr()


def prepare_numeric_data(df, columns):
    """Impute missing values and standardize numeric features."""

    if not columns:
        raise ValueError("Select at least one numeric feature.")

    data = df[columns].replace([np.inf, -np.inf], np.nan)

    # Features with no usable values cannot be imputed.
    usable_columns = [
        column
        for column in columns
        if data[column].notna().any()
    ]

    if not usable_columns:
        raise ValueError(
            "The selected features contain no usable numeric values."
        )

    data = data[usable_columns]

    # Remove constant columns because they cannot help distinguish groups.
    varying_columns = [
        column
        for column in data.columns
        if data[column].nunique(dropna=True) > 1
    ]

    if not varying_columns:
        raise ValueError(
            "The selected features do not contain enough variation."
        )

    data = data[varying_columns]

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(data)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    return data, scaled


def cluster_customers(df, columns, number_of_clusters=3):
    """Group customers using K-means clustering."""

    if len(df) < 3:
        raise ValueError(
            "At least three customer records are required."
        )

    if not 2 <= number_of_clusters <= 10:
        raise ValueError(
            "Choose between 2 and 10 clusters."
        )

    if number_of_clusters >= len(df):
        raise ValueError(
            "The number of clusters must be smaller than the row count."
        )

    _, scaled = prepare_numeric_data(df, columns)

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(scaled)

    result = df.copy()
    result["Customer Cluster"] = [
        f"Cluster {label + 1}" for label in labels
    ]

    return result


def detect_anomalies(df, columns, contamination=0.05):
    """Flag potentially unusual records with Isolation Forest."""

    if len(df) < 10:
        raise ValueError(
            "At least 10 records are required for this demonstration."
        )

    if not 0.01 <= contamination <= 0.20:
        raise ValueError(
            "The anomaly proportion must be between 1% and 20%."
        )

    _, scaled = prepare_numeric_data(df, columns)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )

    predictions = model.fit_predict(scaled)
    scores = model.decision_function(scaled)

    result = df.copy()
    result["Anomaly Status"] = np.where(
        predictions == -1,
        "Potentially unusual",
        "Typical",
    )
    result["Anomaly Score"] = scores

    return result


def get_underrepresented_categories(df, minimum_percentage=5.0):
    """Identify categories below a chosen percentage of all records."""

    results = []

    for column in get_categorical_columns(df):
        counts = (
            df[column]
            .fillna("(Missing)")
            .astype(str)
            .value_counts()
        )

        for category, count in counts.items():
            percentage = count / len(df) * 100

            if percentage < minimum_percentage:
                results.append(
                    {
                        "Feature": column,
                        "Category": category,
                        "Count": int(count),
                        "Percentage": round(percentage, 2),
                    }
                )

    return pd.DataFrame(results)
