# standard library imports
import os

# third-party imports
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
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
        ).drop(columns=["EVENT_ID", "DATE", "BOUT_ORDINAL"])
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

    def get_model_probs(self) -> pd.DataFrame:
        """
        Get red and blue fighter probabilities for all models
        """

        model_probs = pd.DataFrame()
        model_probs["BOUT_ID"] = self.test_df["BOUT_ID"].copy()

        X_test = self.test_df.drop(columns=["BOUT_ID", "RED_WIN"])

        # Logistic Regression
        lr_probs = self.lr.predict_proba(X_test)
        red_probs_lr, blue_probs_lr = lr_probs[:, 1], lr_probs[:, 0]
        model_probs["RED_PROBS_LR"] = red_probs_lr
        model_probs["BLUE_PROBS_LR"] = blue_probs_lr

        # Random Forest
        rf_probs = self.rf.predict_proba(X_test)
        red_probs_rf, blue_probs_rf = rf_probs[:, 1], rf_probs[:, 0]
        model_probs["RED_PROBS_RF"] = red_probs_rf
        model_probs["BLUE_PROBS_RF"] = blue_probs_rf

        # Gradient Boosting
        gbm_probs = self.gbm.predict_proba(X_test)
        red_probs_gbm, blue_probs_gbm = gbm_probs[:, 1], gbm_probs[:, 0]
        model_probs["RED_PROBS_GBM"] = red_probs_gbm
        model_probs["BLUE_PROBS_GBM"] = blue_probs_gbm

        # Bookmaker Odds
        red_probs_vig = 1 / self.odds_df["RED_FIGHTER_ODDS"]
        blue_probs_vig = 1 / self.odds_df["BLUE_FIGHTER_ODDS"]
        model_probs["RED_PROBS_BOOKIES"] = red_probs_vig / (
            red_probs_vig + blue_probs_vig
        )
        model_probs["BLUE_PROBS_BOOKIES"] = blue_probs_vig / (
            red_probs_vig + blue_probs_vig
        )

        return model_probs

    def get_metrics(self, model_probs: pd.DataFrame) -> None:
        """
        Compute log loss and Brier score for all models and
        bookmaker odds
        """

        metrics_dict = {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "Gradient Boosting",
                "Bookmakers",
            ],
            "Log Loss": [
                log_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_LR"],
                ),
                log_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_RF"],
                ),
                log_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_GBM"],
                ),
                log_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_BOOKIES"],
                ),
            ],
            "Brier Score": [
                brier_score_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_LR"],
                ),
                brier_score_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_RF"],
                ),
                brier_score_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_GBM"],
                ),
                brier_score_loss(
                    self.test_df["RED_WIN"],
                    model_probs["RED_PROBS_BOOKIES"],
                ),
            ],
        }

        metrics_df = pd.DataFrame(metrics_dict)
        display(metrics_df)

    def plot_calibration(self, model_probs: pd.DataFrame) -> None:
        """
        Plot calibration curves for all models and bookmaker odds
        """

        plt.style.use("ggplot")
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        ax[0, 0] = CalibrationDisplay.from_predictions(
            self.test_df["RED_WIN"],
            model_probs["RED_PROBS_LR"],
            ax=ax[0, 0],
            name="Logistic Regression",
            color="#ff8389",
            n_bins=10,
        )
        ax[0, 1] = CalibrationDisplay.from_predictions(
            self.test_df["RED_WIN"],
            model_probs["RED_PROBS_RF"],
            ax=ax[0, 1],
            name="Random Forest",
            color="#6929c4",
            n_bins=10,
        )
        ax[1, 0] = CalibrationDisplay.from_predictions(
            self.test_df["RED_WIN"],
            model_probs["RED_PROBS_GBM"],
            ax=ax[1, 0],
            name="Gradient Boosting",
            color="#1192e8",
            n_bins=10,
        )
        ax[1, 1] = CalibrationDisplay.from_predictions(
            self.test_df["RED_WIN"],
            model_probs["RED_PROBS_BOOKIES"],
            ax=ax[1, 1],
            name="Bookmakers",
            color="grey",
            n_bins=10,
        )

        plt.tight_layout()
        plt.show()

    def __call__(self) -> None:
        model_probs = self.get_model_probs()
        self.get_metrics(model_probs)
        self.plot_calibration(model_probs)
