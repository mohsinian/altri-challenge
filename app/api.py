from flask import Flask, request, jsonify, render_template
import os
import logging
from werkzeug.utils import secure_filename
from app.scoring import PropertyScorer
from app.models import PropertyModels
from app.utils import load_data, validate_data

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global cache for scored properties
_scored_properties_cache = None
_cache_timestamp = None

app_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
app = Flask(
    __name__,
    template_folder=os.path.join(parent_dir, "web"),
    static_folder=os.path.join(parent_dir, "web", "static"),
)

# Use absolute paths for data and models folders
app.config["UPLOAD_FOLDER"] = os.path.join(parent_dir, "data")
app.config["MODELS_FOLDER"] = os.path.join(parent_dir, "models")

# Initialize scorer
scorer = PropertyScorer()


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def train_models():
    """Train ML models using sold properties data"""
    try:
        # Load sold properties data
        sold_df = load_data(
            os.path.join(parent_dir, "data", "sold_properties.csv"), is_sold=True
        )

        # Validate data
        if not validate_data(sold_df, is_sold=True):
            return jsonify({"error": "Invalid sold properties data"}), 400

        # Train models
        models = PropertyModels()
        resale_model, resale_metrics = models.train_resale_model(sold_df)
        renovation_model, renovation_metrics = models.train_renovation_model(sold_df)

        # Save models
        if not os.path.exists(app.config["MODELS_FOLDER"]):
            os.makedirs(app.config["MODELS_FOLDER"])

        models.save_models(app.config["MODELS_FOLDER"])

        # Update scorer with new models
        scorer.models = models
        
        # Clear the cache since models have been updated
        global _scored_properties_cache, _cache_timestamp
        _scored_properties_cache = None
        _cache_timestamp = None
        logger.info("API: Cache cleared after model training")

        return jsonify(
            {
                "message": "Models trained successfully",
                "resale_model_performance": {
                    "model_type": type(resale_model.named_steps["regressor"]).__name__,
                    "best_model": type(resale_model.named_steps["regressor"]).__name__,
                    "all_models_comparison": resale_metrics,
                },
                "renovation_model_performance": {
                    "model_type": type(
                        renovation_model.named_steps["regressor"]
                    ).__name__,
                    "best_model": type(renovation_model.named_steps["regressor"]).__name__,
                    "all_models_comparison": renovation_metrics,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/score", methods=["POST"])
def score_properties():
    """Score for-sale properties for flipping potential"""
    global _scored_properties_cache, _cache_timestamp
    
    try:
        logger.info("API: /api/score endpoint called")
        logger.debug("Received request to /api/score endpoint")

        # Check if models are trained
        logger.debug("Checking if models are trained...")
        if scorer.models.resale_model is None or scorer.models.renovation_model is None:
            logger.error("Models not trained")
            return jsonify(
                {"error": "Models not trained. Please train models first."}
            ), 400

        # Check if we have cached results
        if _scored_properties_cache is not None:
            logger.info("API: Returning cached scored properties")
            # Log sample property structure from cache
            if _scored_properties_cache and len(_scored_properties_cache) > 0:
                sample_property = _scored_properties_cache[0]
                logger.info(f"Sample cached property keys: {list(sample_property.keys())}")
                logger.info(f"Sample cached property neighborhood value: {sample_property.get('neighborhoods', 'NOT_FOUND')}")
            return jsonify({"properties": _scored_properties_cache, "count": len(_scored_properties_cache), "cached": True})

        # Load for-sale properties data
        logger.info("API: No cache found, scoring properties (this will trigger NLP processing)")
        logger.debug("Loading for-sale properties data...")
        for_sale_df = load_data(
            os.path.join(parent_dir, "data", "for_sale_properties.csv"), is_sold=False
        )
        logger.debug(f"Loaded data with shape: {for_sale_df.shape}")
        
        # Log unique neighborhoods in the data
        if "neighborhoods" in for_sale_df.columns:
            unique_neighborhoods = for_sale_df["neighborhoods"].dropna().unique().tolist()
            logger.info(f"Unique neighborhoods in data: {unique_neighborhoods}")

        # Validate data
        logger.debug("Validating data...")
        if not validate_data(for_sale_df, is_sold=False):
            logger.error("Data validation failed")
            return jsonify({"error": "Invalid for-sale properties data"}), 400

        # Score properties
        logger.debug("Scoring properties...")
        results = scorer.score_properties(for_sale_df)
        logger.debug(f"Scoring completed. Results count: {len(results)}")
        
        # Log sample property structure
        if results and len(results) > 0:
            sample_property = results[0]
            logger.info(f"Sample property keys: {list(sample_property.keys())}")
            logger.info(f"Sample property neighborhood value: {sample_property.get('neighborhoods', 'NOT_FOUND')}")
        
        # Cache the results
        _scored_properties_cache = results
        import time
        _cache_timestamp = time.time()
        logger.info(f"API: Results cached at timestamp {_cache_timestamp}")

        return jsonify({"properties": results, "count": len(results), "cached": False})
    except Exception as e:
        logger.error(f"Error in /api/score: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/score/upload", methods=["POST"])
def score_uploaded_properties():
    """Score uploaded for-sale properties"""
    try:
        # Check if models are trained
        if scorer.models.resale_model is None or scorer.models.renovation_model is None:
            return jsonify(
                {"error": "Models not trained. Please train models first."}
            ), 400

        # Check if file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Load uploaded data
            for_sale_df = load_data(filepath, is_sold=False)

            # Validate data
            if not validate_data(for_sale_df, is_sold=False):
                return jsonify({"error": "Invalid for-sale properties data"}), 400

            # Score properties
            results = scorer.score_properties(for_sale_df)

            return jsonify({"properties": results, "count": len(results)})
        else:
            return jsonify({"error": "Only CSV files are supported"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/property/<property_id>", methods=["GET"])
def get_property_score(property_id):
    """Get score for a specific property"""
    try:
        # Check if models are trained
        if scorer.models.resale_model is None or scorer.models.renovation_model is None:
            return jsonify(
                {"error": "Models not trained. Please train models first."}
            ), 400

        # Load for-sale properties data
        for_sale_df = load_data(
            os.path.join(parent_dir, "data", "for_sale_properties.csv"), is_sold=False
        )

        # Find property by ID
        property_data = for_sale_df[for_sale_df["property_id"] == property_id]

        if property_data.empty:
            return jsonify({"error": "Property not found"}), 404

        # Score property
        results = scorer.score_properties(property_data)

        if not results:
            return jsonify({"error": "Could not score property"}), 500

        return jsonify(results[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/filters", methods=["GET"])
def get_filter_options():
    """Get filter options for the UI"""
    try:
        # Load for-sale properties data
        for_sale_df = load_data(
            os.path.join(parent_dir, "data", "for_sale_properties.csv"), is_sold=False
        )
        logger.debug(f"Data loaded with shape: {for_sale_df.shape}")
        logger.debug(f"Columns in dataframe: {for_sale_df.columns.tolist()}")

        # Extract filter options
        if "neighborhoods" in for_sale_df.columns:
            neighborhoods = for_sale_df["neighborhoods"].dropna().unique().tolist()
            logger.debug(
                f"Found {len(neighborhoods)} neighborhoods: {neighborhoods[:5]}..."
            )
        else:
            logger.error("Column 'neighborhoods' not found in dataframe")
            neighborhoods = []

        filters = {
            "min_price": int(for_sale_df["list_price"].min()),
            "max_price": int(for_sale_df["list_price"].max()),
            "min_beds": int(for_sale_df["beds"].min()),
            "max_beds": int(for_sale_df["beds"].max()),
            "min_baths": int(for_sale_df["full_baths"].min()),
            "max_baths": int(for_sale_df["full_baths"].max()),
            "neighborhoods": neighborhoods,
            "zip_codes": for_sale_df["zip_code"].dropna().unique().tolist(),
        }

        logger.debug(f"Returning filters: {filters}")

        return jsonify(filters)
    except Exception as e:
        logger.error(f"Error in /api/filters: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the scored properties cache"""
    global _scored_properties_cache, _cache_timestamp
    try:
        logger.info("API: Clearing scored properties cache")
        _scored_properties_cache = None
        _cache_timestamp = None
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
