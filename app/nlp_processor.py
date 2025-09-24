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

# Download NLTK resources with error handling
try:
    logger.info("Downloading NLTK resources...")
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)  # Add the missing resource
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_pipeline = pipeline("sentiment-analysis")

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
