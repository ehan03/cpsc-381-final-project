# standard library imports
import os

# third-party imports
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# local imports


class ModelTrainer:
    """
    All-purpose class for training the models used in this project
    """

    DIR_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(DIR_PATH, "..", "..", "data")
    MODELS_PATH = os.path.join(DIR_PATH, "..", "..", "models")

    def __init__(self, model_name: str) -> None:
        if model_name not in ["logistic_regression", "random_forest", "lightgbm"]:
            raise ValueError(
                "Invalid model name. Please choose one of 'logistic_regression', 'random_forest', or 'lightgbm'."
            )

        self.model_name = model_name
