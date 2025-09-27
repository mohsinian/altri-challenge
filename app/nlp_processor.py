import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import pipeline
import warnings
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Global variables for caching NLP resources
_nltk_resources_downloaded = False
_spacy_model_loaded = None
_nlp_processor_instance = None

def download_nltk_resources():
    """Download NLTK resources only once"""
    global _nltk_resources_downloaded
    if not _nltk_resources_downloaded:
        logger.info("MODULE LEVEL: Downloading NLTK resources...")
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt_tab", quiet=True)  # Add the missing resource
            _nltk_resources_downloaded = True
            logger.info("MODULE LEVEL: NLTK resources downloaded successfully")
        except Exception as e:
            logger.error(f"MODULE LEVEL: Error downloading NLTK resources: {e}")
    else:
        logger.info("MODULE LEVEL: NLTK resources already downloaded, skipping")

def get_spacy_model():
    """Load spaCy model only once"""
    global _spacy_model_loaded
    if _spacy_model_loaded is None:
        logger.info("MODULE LEVEL: Loading spaCy model...")
        try:
            _spacy_model_loaded = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("MODULE LEVEL: spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"MODULE LEVEL: Error loading spaCy model: {e}")
            raise
    else:
        logger.info("MODULE LEVEL: spaCy model already loaded, reusing")
    return _spacy_model_loaded

# Initialize resources
download_nltk_resources()
nlp = get_spacy_model()


def get_nlp_processor():
    """Get a singleton instance of NLPProcessor"""
    global _nlp_processor_instance
    if _nlp_processor_instance is None:
        logger.info("NLPProcessor SINGLETON: Creating new NLPProcessor instance")
        _nlp_processor_instance = NLPProcessor()
        logger.info("NLPProcessor SINGLETON: NLPProcessor instance created and cached")
    else:
        logger.info("NLPProcessor SINGLETON: Reusing existing NLPProcessor instance")
    return _nlp_processor_instance

class NLPProcessor:
    def __init__(self):
        logger.info("NLPProcessor INSTANCE: Creating new NLPProcessor instance")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        logger.info("NLPProcessor INSTANCE: NLPProcessor instance created successfully")

        # Renovation keywords
        self.renovation_keywords = {
            "high": [
                "major renovation",
                "complete rehab",
                "structural issues",
                "foundation problems",
                "water damage",
                "mold",
                "roof replacement",
                "electrical issues",
                "plumbing issues",
            ],
            "medium": [
                "needs updating",
                "outdated",
                "cosmetic updates",
                "kitchen remodel",
                "bathroom remodel",
                "flooring",
                "paint",
                "fixtures",
                "some repairs",
            ],
            "low": [
                "move-in ready",
                "recently updated",
                "renovated",
                "modern",
                "new appliances",
                "well-maintained",
                "impeccable condition",
                "turnkey",
            ],
        }

        # Positive features that increase value
        self.value_adding_features = [
            "granite",
            "stainless steel",
            "hardwood floors",
            "updated kitchen",
            "renovated bathroom",
            "new roof",
            "new windows",
            "fenced yard",
            "garage",
            "patio",
            "deck",
            "basement",
            "natural light",
            "open concept",
            "energy efficient",
            "modern",
            "spacious",
        ]

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize with error handling and logging
        try:
            logger.debug(f"Tokenizing text: {text[:50]}...")
            tokens = word_tokenize(text)
            logger.debug(f"Tokenization successful, got {len(tokens)} tokens")
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            # Fallback tokenization if NLTK fails
            tokens = text.split()
            logger.debug(f"Used fallback tokenization, got {len(tokens)} tokens")

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]

        return " ".join(tokens)

    def extract_renovation_level(self, text):
        """Extract renovation level from property description"""
        if pd.isna(text):
            return "medium"  # Default assumption

        text_lower = text.lower()

        # Check for high renovation needs
        for keyword in self.renovation_keywords["high"]:
            if keyword in text_lower:
                return "high"

        # Check for low renovation needs
        for keyword in self.renovation_keywords["low"]:
            if keyword in text_lower:
                return "low"

        # Check for medium renovation needs
        for keyword in self.renovation_keywords["medium"]:
            if keyword in text_lower:
                return "medium"

        # Default to medium if no keywords found
        return "medium"

    def count_value_adding_features(self, text):
        """Count the number of value-adding features mentioned in the description"""
        if pd.isna(text):
            return 0

        text_lower = text.lower()
        count = 0

        for feature in self.value_adding_features:
            if feature in text_lower:
                count += 1

        return count

    def get_sentiment_score(self, text):
        """Get sentiment score of property description"""
        if pd.isna(text):
            return 0

        try:
            # Truncate text if too long (model limit)
            if len(text) > 512:
                text = text[:512]

            result = self.sentiment_pipeline(text)[0]
            # Convert to a score between -1 and 1
            if result["label"] == "POSITIVE":
                return result["score"]
            else:
                return -result["score"]
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"Error in sentiment analysis: {e}")
            return 0


    def extract_features(self, df):
        """Extract NLP features from property descriptions"""
        df_copy = df.copy()

        # Preprocess text
        df_copy["processed_text"] = df_copy["text"].apply(self.preprocess_text)

        # Extract renovation level
        df_copy["renovation_level"] = df_copy["text"].apply(
            self.extract_renovation_level
        )

        # Count value-adding features
        df_copy["value_features_count"] = df_copy["text"].apply(
            self.count_value_adding_features
        )

        # Get sentiment score
        df_copy["sentiment_score"] = df_copy["text"].apply(self.get_sentiment_score)

        # Convert renovation level to numeric
        renovation_mapping = {"low": 1, "medium": 2, "high": 3}
        df_copy["renovation_level_numeric"] = df_copy["renovation_level"].map(
            renovation_mapping
        )

        return df_copy
