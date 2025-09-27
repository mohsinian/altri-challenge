import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb


def load_data(filepath, is_sold=True):
    """Load data from CSV file and filter out rows with missing critical values"""
    try:
        print(f"Attempting to load data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from: {filepath}")

        # Log initial data shape and missing values
        print(f"Initial data shape: {df.shape}")
        print(f"Columns in dataframe: {df.columns.tolist()}")

        if "sqft" in df.columns:
            print(f"Missing sqft values: {df['sqft'].isnull().sum()}")
        else:
            print("Column 'sqft' not found in dataframe")

        # Define critical columns that must have values
        critical_columns = ["property_id", "sqft"]
        if is_sold:
            critical_columns.append("sold_price")
        else:
            critical_columns.append("list_price")

        # Log missing values for all critical columns
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                print(f"Missing {col} values: {missing_count}")
            else:
                print(f"Column '{col}' not found in dataframe")

        # Filter out rows with missing values in any critical column
        initial_rows = len(df)
        df = df.dropna(subset=critical_columns)
        filtered_rows = len(df)

        print(
            f"Filtered out {initial_rows - filtered_rows} rows with missing critical values"
        )
        print(f"Final data shape: {df.shape}")

        return df
    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
        raise
    except Exception as e:
        print(f"Error loading data from {filepath}: {str(e)}")
        raise


def validate_data(df, is_sold=True):
    """Validate data format and required columns"""
    required_columns = [
        "property_id",
        "property_url",
        "formatted_address",
        "city",
        "state",
        "zip_code",
        "beds",
        "full_baths",
        "sqft",
        "year_built",
    ]

    if is_sold:
        required_columns.append("sold_price")
    else:
        required_columns.append("list_price")

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False

    # Check for missing values in critical columns
    critical_columns = ["property_id", "sqft"]
    if is_sold:
        critical_columns.append("sold_price")
    else:
        critical_columns.append("list_price")

    for col in critical_columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(
                f"Missing values in critical column: {col} - {missing_count} rows affected"
            )
            print(f"Total rows before filtering: {len(df)}")
            return False

    # Check for valid values
    if is_sold:
        if (df["sold_price"] <= 0).any():
            print("Invalid sold price values")
            return False
    else:
        if (df["list_price"] <= 0).any():
            print("Invalid list price values")
            return False

    if (df["sqft"] <= 0).any():
        print("Invalid square footage values")
        return False

    return True


def clean_data(df):
    """Clean and preprocess data"""
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # Fill missing values
    df_clean["beds"] = df_clean["beds"].fillna(0)
    df_clean["full_baths"] = df_clean["full_baths"].fillna(0)
    df_clean["half_baths"] = df_clean["half_baths"].fillna(0)
    df_clean["year_built"] = df_clean["year_built"].fillna(1900)

    # Fill missing lot_sqft with median value
    if "lot_sqft" in df_clean.columns:
        median_lot = df_clean["lot_sqft"].median()
        df_clean["lot_sqft"] = df_clean["lot_sqft"].fillna(median_lot)

    # Fill missing stories with 1
    if "stories" in df_clean.columns:
        df_clean["stories"] = df_clean["stories"].fillna(1)

    return df_clean


def create_model_pipelines(preprocessor):
    """Create a dictionary of model pipelines with the given preprocessor"""
    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(random_state=42)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", GradientBoostingRegressor(random_state=42)),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", xgb.XGBRegressor(random_state=42)),
            ]
        ),
    }
    return models


def preprocess_for_model(df, is_sold=True):
    """Preprocess data for model training or prediction"""
    # Clean data
    df_clean = clean_data(df)

    # Calculate price per square foot
    if is_sold:
        df_clean["price_per_sqft"] = df_clean["sold_price"] / df_clean["sqft"]
    else:
        df_clean["price_per_sqft"] = df_clean["list_price"] / df_clean["sqft"]

    # Calculate property age
    current_year = pd.Timestamp.now().year
    df_clean["property_age"] = current_year - df_clean["year_built"]

    # Fill missing text with empty string
    df_clean["text"] = df_clean["text"].fillna("")

    return df_clean
