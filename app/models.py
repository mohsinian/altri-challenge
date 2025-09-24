import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from app.nlp_processor import NLPProcessor


class PropertyModels:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.resale_model = None
        self.renovation_model = None
        self.preprocessor = None
        self.feature_columns = None

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

        # Define models to try
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

        # Evaluate models
        best_model = None
        best_score = float("inf")
        best_name = ""

        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
            )
            avg_rmse = np.sqrt(-cv_scores.mean())

            print(f"{name}: RMSE = {avg_rmse:.2f}")

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_model = model
                best_name = name

        print(f"Best model: {best_name} with RMSE: {best_score:.2f}")

        # Fit the best model
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

        return best_model

    def train_renovation_model(self, sold_df):
        """Train model to estimate renovation costs"""
        print("Training renovation cost estimation model...")

        # Create a synthetic renovation cost based on property characteristics
        # In a real scenario, you would have actual renovation cost data
        df = self.nlp_processor.extract_features(sold_df.copy())

        # Base renovation cost per square foot
        base_cost_per_sqft = {
            "low": 10,  # $10 per sqft for cosmetic updates
            "medium": 25,  # $25 per sqft for moderate renovations
            "high": 50,  # $50 per sqft for major renovations
        }

        # Calculate renovation cost
        df["renovation_cost"] = df.apply(
            lambda row: base_cost_per_sqft.get(row["renovation_level"], 25)
            * row["sqft"],
            axis=1,
        )

        # Add some randomness to simulate real-world variation
        df["renovation_cost"] = df["renovation_cost"] * np.random.normal(
            1.0, 0.2, len(df)
        )

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

        # Define model
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(random_state=42)),
            ]
        )

        # Fit model
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        print(f"Renovation Cost Model - Test RMSE: ${test_rmse:.2f}")
        print(f"Renovation Cost Model - Test R²: {test_r2:.2f}")

        # Store the model
        self.renovation_model = model

        return model

    def predict_resale_value(self, property_df):
        """Predict after-renovation resale value"""
        if self.resale_model is None:
            raise ValueError("Resale model not trained. Call train_resale_model first.")

        # Prepare features
        X, _ = self.prepare_features(property_df, is_sold=False)

        # Make predictions
        predictions = self.resale_model.predict(X)

        return predictions

    def predict_renovation_cost(self, property_df):
        """Predict renovation cost"""
        if self.renovation_model is None:
            raise ValueError(
                "Renovation model not trained. Call train_renovation_model first."
            )

        # Extract NLP features
        df = self.nlp_processor.extract_features(property_df.copy())

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
