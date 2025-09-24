import pandas as pd
from datetime import datetime
from app.models import PropertyModels


class PropertyScorer:
    def __init__(self, models_path=None):
        self.models = PropertyModels()

        if models_path:
            self.models.load_models(models_path)

        # Define constants for calculations
        self.carrying_cost_rate = 0.02  # 2% of property value for 6 months
        self.selling_cost_rate = (
            0.06  # 6% for real estate agent fees, closing costs, etc.
        )
        self.contingency_rate = 0.1  # 10% contingency for unexpected costs

        # Risk factors
        self.risk_factors = {
            "age": {
                "threshold": 50,
                "weight": 0.3,
            },  # Properties older than 50 years have higher risk
            "days_on_market": {
                "threshold": 60,
                "weight": 0.4,
            },  # Properties on market > 60 days have higher risk
            "renovation": {
                "high": 0.5,
                "medium": 0.3,
                "low": 0.1,
            },  # Renovation level risk weights
        }

    def calculate_carrying_costs(self, property_value, months=6):
        """Calculate carrying costs for a property"""
        annual_rate = (
            0.05  # 5% annual rate (property taxes, insurance, utilities, etc.)
        )
        monthly_rate = annual_rate / 12
        return property_value * monthly_rate * months

    def calculate_selling_costs(self, sale_price):
        """Calculate selling costs (agent fees, closing costs, etc.)"""
        return sale_price * self.selling_cost_rate

    def calculate_contingency(self, renovation_cost):
        """Calculate contingency for unexpected renovation costs"""
        return renovation_cost * self.contingency_rate

    def calculate_profit(self, purchase_price, resale_value, renovation_cost):
        """Calculate expected profit from a flip"""
        carrying_costs = self.calculate_carrying_costs(purchase_price)
        selling_costs = self.calculate_selling_costs(resale_value)
        contingency = self.calculate_contingency(renovation_cost)

        total_costs = (
            purchase_price
            + renovation_cost
            + carrying_costs
            + selling_costs
            + contingency
        )
        profit = resale_value - total_costs

        return {
            "profit": profit,
            "carrying_costs": carrying_costs,
            "selling_costs": selling_costs,
            "contingency": contingency,
            "total_costs": total_costs,
        }

    def calculate_roi(self, profit, total_investment):
        """Calculate Return on Investment"""
        if total_investment == 0:
            return 0
        return (profit / total_investment) * 100

    def calculate_risk_score(self, property_df):
        """Calculate risk score for a property"""
        risk_score = 0

        # Age risk
        current_year = datetime.now().year
        property_age = current_year - property_df["year_built"].iloc[0]

        if property_age > self.risk_factors["age"]["threshold"]:
            age_risk = min(
                (property_age - self.risk_factors["age"]["threshold"]) / 50, 1
            )  # Cap at 1
            risk_score += age_risk * self.risk_factors["age"]["weight"]

        # Days on market risk
        if "days_on_mls" in property_df.columns:
            days_on_market = property_df["days_on_mls"].iloc[0]
            if (
                not pd.isna(days_on_market)
                and days_on_market > self.risk_factors["days_on_market"]["threshold"]
            ):
                market_risk = min(
                    (days_on_market - self.risk_factors["days_on_market"]["threshold"])
                    / 90,
                    1,
                )  # Cap at 1
                risk_score += (
                    market_risk * self.risk_factors["days_on_market"]["weight"]
                )

        # Renovation risk
        if "renovation_level" in property_df.columns:
            renovation_level = property_df["renovation_level"].iloc[0]
            if renovation_level in self.risk_factors["renovation"]:
                risk_score += self.risk_factors["renovation"][renovation_level]

        # Normalize risk score to 0-100
        risk_score = min(risk_score * 100, 100)

        return risk_score

    def assign_grade(self, roi, risk_score):
        """Assign letter grade based on ROI and risk"""
        # High ROI, Low Risk
        if roi > 30 and risk_score < 30:
            return "A"
        # Medium ROI, Low Risk or High ROI, Medium Risk
        elif (roi > 15 and risk_score < 30) or (roi > 30 and risk_score < 60):
            return "B"
        # Low ROI, Low Risk or Medium ROI, Medium Risk
        elif (roi > 5 and risk_score < 30) or (roi > 15 and risk_score < 60):
            return "C"
        # Low ROI, Medium Risk or Medium ROI, High Risk
        elif (roi > 0 and risk_score < 60) or (roi > 15 and risk_score >= 60):
            return "D"
        # Negative ROI or High Risk
        else:
            return "F"

    def score_properties(self, for_sale_df):
        """Score for-sale properties for flipping potential"""
        # Make a copy to avoid modifying the original dataframe
        df = for_sale_df.copy()

        # Calculate price per square foot
        df["price_per_sqft"] = df["list_price"] / df["sqft"]

        # Extract NLP features
        from app.nlp_processor import NLPProcessor

        nlp_processor = NLPProcessor()
        df = nlp_processor.extract_features(df)

        # Predict resale value
        resale_values = self.models.predict_resale_value(df)
        df["predicted_resale_value"] = resale_values

        # Predict renovation cost
        renovation_costs = self.models.predict_renovation_cost(df)
        df["predicted_renovation_cost"] = renovation_costs

        # Calculate profit metrics for each property
        results = []

        for i, property_data in df.iterrows():
            purchase_price = property_data["list_price"]
            resale_value = property_data["predicted_resale_value"]
            renovation_cost = property_data["predicted_renovation_cost"]

            # Calculate profit
            profit_metrics = self.calculate_profit(
                purchase_price, resale_value, renovation_cost
            )

            # Calculate ROI
            total_investment = profit_metrics["total_costs"]
            roi = self.calculate_roi(profit_metrics["profit"], total_investment)

            # Calculate risk score
            property_subset = df.iloc[[i]]
            risk_score = self.calculate_risk_score(property_subset)

            # Assign grade
            grade = self.assign_grade(roi, risk_score)

            # Create result dictionary
            result = {
                "property_id": property_data.get("property_id", ""),
                "property_url": property_data.get("property_url", ""),
                "address": property_data.get("formatted_address", ""),
                "list_price": purchase_price,
                "predicted_resale_value": resale_value,
                "predicted_renovation_cost": renovation_cost,
                "profit": profit_metrics["profit"],
                "carrying_costs": profit_metrics["carrying_costs"],
                "selling_costs": profit_metrics["selling_costs"],
                "contingency": profit_metrics["contingency"],
                "total_costs": profit_metrics["total_costs"],
                "roi": roi,
                "risk_score": risk_score,
                "grade": grade,
                "renovation_level": property_data.get("renovation_level", "medium"),
                "value_features_count": property_data.get("value_features_count", 0),
                "sentiment_score": property_data.get("sentiment_score", 0),
                "beds": property_data.get("beds", 0),
                "baths": property_data.get("full_baths", 0),
                "sqft": property_data.get("sqft", 0),
                "year_built": property_data.get("year_built", 0),
                "days_on_mls": property_data.get("days_on_mls", 0),
                "latitude": property_data.get("latitude", 0),
                "longitude": property_data.get("longitude", 0),
                "primary_photo": property_data.get("primary_photo", ""),
                "neighborhood": property_data.get("neighborhoods", ""),
                "explanation": self._generate_explanation(
                    roi, risk_score, grade, property_data
                ),
            }

            results.append(result)

        return results

    def _generate_explanation(self, roi, risk_score, grade, property_data):
        """Generate explanation for the property score"""
        explanations = []

        # Grade explanation
        grade_explanations = {
            "A": "Excellent investment opportunity with high expected returns and low risk.",
            "B": "Good investment opportunity with solid returns and manageable risk.",
            "C": "Average investment opportunity with moderate returns and risk.",
            "D": "Below average investment opportunity with low returns and/or high risk.",
            "F": "Poor investment opportunity with negative returns and/or very high risk.",
        }
        explanations.append(grade_explanations.get(grade, ""))

        # ROI explanation
        if roi > 30:
            explanations.append(f"Very high expected ROI of {roi:.1f}%.")
        elif roi > 15:
            explanations.append(f"Good expected ROI of {roi:.1f}%.")
        elif roi > 5:
            explanations.append(f"Moderate expected ROI of {roi:.1f}%.")
        elif roi > 0:
            explanations.append(f"Low expected ROI of {roi:.1f}%.")
        else:
            explanations.append(f"Negative expected ROI of {roi:.1f}%.")

        # Risk explanation
        if risk_score < 20:
            explanations.append("Low risk investment.")
        elif risk_score < 40:
            explanations.append("Moderate risk investment.")
        elif risk_score < 60:
            explanations.append("High risk investment.")
        else:
            explanations.append("Very high risk investment.")

        # Property-specific factors
        renovation_level = property_data.get("renovation_level", "medium")
        if renovation_level == "low":
            explanations.append("Property requires minimal renovation.")
        elif renovation_level == "medium":
            explanations.append("Property requires moderate renovation.")
        else:
            explanations.append("Property requires extensive renovation.")

        # Days on market
        days_on_mls = property_data.get("days_on_mls", 0)
        if days_on_mls > 60:
            explanations.append(
                f"Property has been on the market for {days_on_mls} days, which may indicate negotiation opportunities."
            )

        return " ".join(explanations)
