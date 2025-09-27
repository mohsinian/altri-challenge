import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from app.utils import create_model_pipelines


class PropertyModels:
    def __init__(self):
        import logging

        logger = logging.getLogger(__name__)
        logger.info("MODELS: Getting NLPProcessor instance in PropertyModels")
        from app.nlp_processor import get_nlp_processor

        self.nlp_processor = get_nlp_processor()
        self.resale_model = None
        self.renovation_model = None
        self.preprocessor = None
        self.feature_columns = None
        logger.info("MODELS: PropertyModels initialized with NLPProcessor")

    def prepare_features(self, df, is_sold=True):
        """Prepare features for model training or prediction"""
        # Extract NLP features
        df = self.nlp_processor.extract_features(df)

        # Select features
        if is_sold:
            # For sold properties, we have more features available
            features = [
                "sqft",
                "year_built",
                "beds",
                "full_baths",
                "half_baths",
                "lot_sqft",
                "stories",
                "price_per_sqft",
                "renovation_level_numeric",
                "value_features_count",
                "sentiment_score",
            ]

            # Add location-based features
            location_features = ["neighborhoods", "zip_code"]

            # Target variable
            target = "sold_price"
        else:
            # For for-sale properties
            features = [
                "sqft",
                "year_built",
                "beds",
                "full_baths",
                "half_baths",
                "lot_sqft",
                "stories",
                "price_per_sqft",
                "renovation_level_numeric",
                "value_features_count",
                "sentiment_score",
            ]

            # Add location-based features
            location_features = ["neighborhoods", "zip_code"]

            # Target variable (for training only)
            target = None

        # Handle missing values
        for feature in features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())

        # Create feature matrix
        X = df[features].copy()

        # Add location features
        for feature in location_features:
            if feature in df.columns:
                X[feature] = df[feature].fillna("Unknown")

        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()

        if is_sold and target in df.columns:
            y = df[target]
            return X, y
        else:
            return X, None

    def train_resale_model(self, sold_df):
        """Train model to predict after-renovation resale value"""
        print("Training resale value prediction model...")

        # Prepare features
        X, y = self.prepare_features(sold_df, is_sold=True)

        # Add neighborhood median price per sqft as a feature
        neighborhood_median_ppsqft = (
            sold_df.groupby("neighborhoods")["price_per_sqft"].median().to_dict()
        )
        X["neighborhood_median_ppsqft"] = X["neighborhoods"].map(
            neighborhood_median_ppsqft
        )
        X["neighborhood_median_ppsqft"] = X["neighborhood_median_ppsqft"].fillna(
            X["neighborhood_median_ppsqft"].median()
        )

        # Add potential upside feature
        X["potential_upside"] = (
            X["neighborhood_median_ppsqft"] - X["price_per_sqft"]
        ) / X["neighborhood_median_ppsqft"]

        # Add renovation premium based on keyword analysis
        X["renovation_premium"] = X["renovation_level_numeric"] * 0.05  # 5% per level

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define preprocessing for numeric and categorical features
        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Create model pipelines using helper function
        models = create_model_pipelines(preprocessor)

        # Evaluate models
        best_model = None
        best_score = float("inf")
        best_name = ""
        model_metrics = {}  # Store metrics for all models

        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
            )
            avg_rmse = np.sqrt(-cv_scores.mean())

            print(f"{name}: RMSE = {avg_rmse:.2f}")

            # Fit the model to get test metrics
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)

            # Store metrics for this model
            model_metrics[name] = {
                "cv_rmse": avg_rmse,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_model = model
                best_name = name

        print(f"Best model: {best_name} with RMSE: {best_score:.2f}")

        # Fit the best model again (it was already fitted above, but this ensures it's properly fitted)
        best_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Test R²: {test_r2:.2f}")

        # Store the model and preprocessor
        self.resale_model = best_model
        self.preprocessor = preprocessor

        # Return both the best model and all model metrics
        return best_model, model_metrics

    def train_renovation_model(self, sold_df):
        """Train model to estimate renovation costs"""
        print("Training renovation cost estimation model...")

        # Create a synthetic renovation cost based on property characteristics
        df = self.nlp_processor.extract_features(sold_df.copy())

        # Data-driven renovation costs based on EDA
        base_cost_percentage = {
            "low": 0.05,  # 8% of property value for cosmetic updates
            "medium": 0.12,  # 18% of property value for moderate renovations
            "high": 0.20,  # 28% of property value for major renovations
        }

        # Get neighborhood median values
        neighborhood_medians = (
            sold_df.groupby("neighborhoods")["sold_price"].median().to_dict()
        )

        # Calculate renovation cost based on neighborhood median value
        df["neighborhood_median"] = df["neighborhoods"].map(neighborhood_medians)
        df["neighborhood_median"] = df["neighborhood_median"].fillna(
            df["neighborhood_median"].median()
        )

        # Calculate renovation cost
        df["renovation_cost"] = df.apply(
            lambda row: base_cost_percentage.get(row["renovation_level"], 0.15)
            * row["neighborhood_median"],
            axis=1,
        )

        # Add some randomness to simulate real-world variation
        df["renovation_cost"] = df["renovation_cost"] * np.random.normal(
            1.0, 0.1, len(df)
        )

        # Ensure renovation costs are positive and reasonable
        df["renovation_cost"] = df["renovation_cost"].clip(lower=5000)  # Minimum $5,000

        # Prepare features
        features = [
            "sqft",
            "year_built",
            "beds",
            "full_baths",
            "half_baths",
            "lot_sqft",
            "stories",
            "renovation_level_numeric",
            "neighborhood_median",
        ]

        X = df[features].copy()
        y = df["renovation_cost"]

        # Handle missing values
        for feature in features:
            if feature in X.columns:
                X[feature] = X[feature].fillna(X[feature].median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define preprocessing
        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, numeric_features)]
        )

        # Create model pipelines using helper function
        models = create_model_pipelines(preprocessor)

        # Evaluate models
        best_model = None
        best_score = float("inf")
        best_name = ""
        model_metrics = {}  # Store metrics for all models

        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
            )
            avg_rmse = np.sqrt(-cv_scores.mean())

            print(f"Renovation {name}: RMSE = ${avg_rmse:.2f}")

            # Fit the model to get test metrics
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)

            # Store metrics for this model
            model_metrics[name] = {
                "cv_rmse": avg_rmse,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_model = model
                best_name = name

        print(f"Best Renovation model: {best_name} with RMSE: ${best_score:.2f}")

        # Fit the best model again (it was already fitted above, but this ensures it's properly fitted)
        best_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        print(f"Renovation Cost Model - Test RMSE: ${test_rmse:.2f}")
        print(f"Renovation Cost Model - Test R²: {test_r2:.2f}")

        # Store the model
        self.renovation_model = best_model

        # Return both the best model and all model metrics
        return best_model, model_metrics

    def predict_resale_value(self, property_df):
        """Predict after-renovation resale value"""
        if self.resale_model is None:
            raise ValueError("Resale model not trained. Call train_resale_model first.")

        # Make a copy to avoid modifying the original dataframe
        df = property_df.copy()

        # NLP features should already be extracted, but check if they exist
        if "renovation_level_numeric" not in df.columns:
            from app.nlp_processor import get_nlp_processor
            import logging

            logger = logging.getLogger(__name__)

            logger.info(
                "MODELS: Getting fallback NLPProcessor instance in predict_resale_value"
            )
            nlp_processor = get_nlp_processor()
            df = nlp_processor.extract_features(df)
            logger.info("MODELS: Fallback NLP feature extraction completed")

        # Calculate price per square foot if not already calculated
        if "price_per_sqft" not in df.columns:
            df["price_per_sqft"] = df["list_price"] / df["sqft"]

        # Add neighborhood median price per sqft as a feature
        # We need to use the same neighborhood medians from training
        # Since we don't have the sold data here, we'll use hardcoded values based on EDA
        neighborhood_median_ppsqft = {
            "Northwest Warren, Heritage Village": 208.01,
            "Northwest Warren": 203.56,
            "Northeast Warren": 191.91,
            "Southwest Warren": 148.67,
            "Southeast Warren": 133.33,
        }

        df["neighborhood_median_ppsqft"] = df["neighborhoods"].map(
            neighborhood_median_ppsqft
        )
        df["neighborhood_median_ppsqft"] = df["neighborhood_median_ppsqft"].fillna(
            165.63
        )  # Overall median

        # Add potential upside feature
        df["potential_upside"] = (
            df["neighborhood_median_ppsqft"] - df["price_per_sqft"]
        ) / df["neighborhood_median_ppsqft"]

        # Add renovation premium based on keyword analysis
        df["renovation_premium"] = df["renovation_level_numeric"] * 0.05  # 5% per level

        # Prepare features
        X, _ = self.prepare_features(df, is_sold=False)

        # Make sure all required features are present
        required_features = [
            "neighborhood_median_ppsqft",
            "potential_upside",
            "renovation_premium",
        ]
        for feature in required_features:
            if feature not in X.columns:
                X[feature] = 0  # Default value if missing

        # Make predictions
        predictions = self.resale_model.predict(X)

        return predictions

    def predict_renovation_cost(self, property_df):
        """Predict renovation cost"""
        if self.renovation_model is None:
            raise ValueError(
                "Renovation model not trained. Call train_renovation_model first."
            )

        # Make a copy to avoid modifying the original dataframe
        df = property_df.copy()

        # NLP features should already be extracted, but check if they exist
        if "renovation_level_numeric" not in df.columns:
            from app.nlp_processor import get_nlp_processor
            import logging

            logger = logging.getLogger(__name__)

            logger.info(
                "MODELS: Getting fallback NLPProcessor instance in predict_renovation_cost"
            )
            nlp_processor = get_nlp_processor()
            df = nlp_processor.extract_features(df)
            logger.info("MODELS: Fallback NLP feature extraction completed")

        # Calculate neighborhood median values (same as used in training)
        neighborhood_medians = {
            "Northwest Warren, Heritage Village": 300000,  # Approximate based on EDA
            "Northwest Warren": 280000,
            "Northeast Warren": 260000,
            "Southwest Warren": 200000,
            "Southeast Warren": 180000,
        }

        df["neighborhood_median"] = df["neighborhoods"].map(neighborhood_medians)
        df["neighborhood_median"] = df["neighborhood_median"].fillna(
            220000
        )  # Overall median

        # Prepare features
        features = [
            "sqft",
            "year_built",
            "beds",
            "full_baths",
            "half_baths",
            "lot_sqft",
            "stories",
            "renovation_level_numeric",
            "neighborhood_median",
        ]

        X = df[features].copy()

        # Handle missing values
        for feature in features:
            if feature in X.columns:
                X[feature] = X[feature].fillna(X[feature].median())

        # Make predictions
        predictions = self.renovation_model.predict(X)

        return predictions

    def save_models(self, path):
        """Save trained models to disk"""
        if self.resale_model is not None:
            joblib.dump(self.resale_model, os.path.join(path, "resale_model.pkl"))

        if self.renovation_model is not None:
            joblib.dump(
                self.renovation_model, os.path.join(path, "renovation_model.pkl")
            )

        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, os.path.join(path, "preprocessor.pkl"))

    def load_models(self, path):
        """Load trained models from disk"""
        resale_path = os.path.join(path, "resale_model.pkl")
        renovation_path = os.path.join(path, "renovation_model.pkl")
        preprocessor_path = os.path.join(path, "preprocessor.pkl")

        if os.path.exists(resale_path):
            self.resale_model = joblib.load(resale_path)

        if os.path.exists(renovation_path):
            self.renovation_model = joblib.load(renovation_path)

        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
