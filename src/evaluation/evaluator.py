# standard library imports
import os

# third-party imports
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss, log_loss

# local imports


class Evaluator:
    """
    Class for evaluating the models on the test set with respect
    to log loss, Brier score, and calibration plots as well as
    compare to the bookmaker odds
    """

    DIR_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(DIR_PATH, "..", "..", "data")
    MODELS_PATH = os.path.join(DIR_PATH, "..", "..", "models")

    def __init__(self) -> None:
        self.test_df = pd.read_csv(
            os.path.join(self.DATA_PATH, "processed", "test.csv")
        ).drop(columns=["EVENT_ID", "DATE", "BOUT_ORDINAL", "RED_WIN"])
        self.odds_df = pd.read_csv(
            os.path.join(self.DATA_PATH, "processed", "backtest_odds.csv")
        )

        # Drop rows where there is no winner (draws or no contests)
        self.test_df = self.test_df.loc[self.test_df["RED_WIN"].notnull()]
        self.odds_df = self.odds_df.loc[self.odds_df["RED_WIN"].notnull()]

        # Load prefit models
        self.lr = load(os.path.join(self.MODELS_PATH, "logistic_regression.joblib"))
        self.rf = load(os.path.join(self.MODELS_PATH, "random_forest.joblib"))
        self.gbm = load(os.path.join(self.MODELS_PATH, "gradient_boosting.joblib"))
