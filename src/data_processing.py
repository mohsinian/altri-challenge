import pandas as pd
from datetime import datetime
from config.settings import FOR_SALE_PROPERTIES_PATH, SOLD_PROPERTIES_PATH


def load_data():
    """Load and preprocess the data"""
    # Load datasets
    for_sale_df = pd.read_csv(FOR_SALE_PROPERTIES_PATH)
    sold_df = pd.read_csv(SOLD_PROPERTIES_PATH)

    # Convert data types
    for_sale_df = convert_data_types(for_sale_df)
    sold_df = convert_data_types(sold_df)

    # Calculate derived features
    for_sale_df = calculate_derived_features(for_sale_df)
    sold_df = calculate_derived_features(sold_df)

    return for_sale_df, sold_df


def convert_data_types(df):
    """Convert relevant columns to appropriate data types"""
    numeric_cols = [
        "list_price",
        "sqft",
        "year_built",
        "beds",
        "full_baths",
        "half_baths",
        "tax",
        "hoa_fee",
        "days_on_mls",
        "estimated_value",
        "assessed_value",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def calculate_derived_features(df):
    """Calculate derived features"""
    # Calculate property age
    current_year = datetime.now().year
    df["property_age"] = current_year - df["year_built"]

    # Calculate price per square foot
    if "list_price" in df.columns and "sqft" in df.columns:
        df["list_price_per_sqft"] = df["list_price"] / df["sqft"]

    if "sold_price" in df.columns and "sqft" in df.columns:
        df["sold_price_per_sqft"] = df["sold_price"] / df["sqft"]

    return df


for_sale_df, sold_df = load_data()
print(for_sale_df)
print(sold_df)