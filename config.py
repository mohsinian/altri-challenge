import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key"
    MODELS_FOLDER = os.environ.get("MODELS_FOLDER") or "../models"
    DATA_FOLDER = os.environ.get("DATA_FOLDER") or "../data"
