# standard library imports
import os
from typing import Union

# third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import load

# local imports
from src.bet_sizing import SimultaneousKelly


class BacktestFramework:
    """
    Class for evaluating betting performance of the models over events
    from 2022 to early 2024
    """

    DIR_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(DIR_PATH, "..", "..", "data")
    MODELS_PATH = os.path.join(DIR_PATH, "..", "..", "models")

    def __init__(self, initial_bankroll: float = 100.0) -> None:
        """
        Initialize the BacktestFramework class
        """

        self.initial_bankroll = initial_bankroll
        self.test_df = pd.read_csv(
            os.path.join(self.DATA_PATH, "processed", "test.csv")
        ).drop(columns=["EVENT_ID", "DATE", "BOUT_ORDINAL", "RED_WIN"])
        self.odds_df = pd.read_csv(
            os.path.join(self.DATA_PATH, "processed", "backtest_odds.csv")
        )

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

        X_test = self.test_df.drop(columns=["BOUT_ID"])

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

        # Ensemble
        red_probs_ensemble = (red_probs_lr + red_probs_rf + red_probs_gbm) / 3
        blue_probs_ensemble = (blue_probs_lr + blue_probs_rf + blue_probs_gbm) / 3
        model_probs["RED_PROBS_ENSEMBLE"] = red_probs_ensemble
        model_probs["BLUE_PROBS_ENSEMBLE"] = blue_probs_ensemble

        return model_probs

    def helper_calculate_profit(
        self, wager_side: str, red_win: Union[int, float], wager: float, odds: float
    ) -> float:
        if np.isnan(red_win) or wager == 0.0:
            return 0.0

        if wager_side == "RED":
            return wager * red_win * odds - wager
        else:
            return wager * (1 - red_win) * odds - wager

    def run_backtest(self, model_probs: pd.DataFrame):
        combined_df = self.odds_df.merge(model_probs, on="BOUT_ID", how="inner")
        event_ids = combined_df["EVENT_ID"].unique()

        results_dict = {
            "DATES": [pd.to_datetime("2022-01-01")],
            "BANKROLL_LR": [self.initial_bankroll],
            "BANKROLL_RF": [self.initial_bankroll],
            "BANKROLL_GBM": [self.initial_bankroll],
            "BANKROLL_ENSEMBLE": [self.initial_bankroll],
            "TOTAL_WAGER_LR": [0.0],
            "TOTAL_WAGER_RF": [0.0],
            "TOTAL_WAGER_GBM": [0.0],
            "TOTAL_WAGER_ENSEMBLE": [0.0],
            "ROI_LR": [0.0],
            "ROI_RF": [0.0],
            "ROI_GBM": [0.0],
            "ROI_ENSEMBLE": [0.0],
        }

        for event_id in event_ids:
            sliced_df = combined_df.loc[combined_df["EVENT_ID"] == event_id].copy()
            red_odds = sliced_df["RED_FIGHTER_ODDS"].to_numpy()
            blue_odds = sliced_df["BLUE_FIGHTER_ODDS"].to_numpy()
            results_dict["DATES"].append(pd.to_datetime(sliced_df["DATE"].values[0]))

            for model in ["LR", "RF", "GBM", "ENSEMBLE"]:
                red_probs = sliced_df[f"RED_PROBS_{model}"].to_numpy()
                blue_probs = sliced_df[f"BLUE_PROBS_{model}"].to_numpy()

                kelly = SimultaneousKelly(
                    red_probs=red_probs,
                    blue_probs=blue_probs,
                    red_odds=red_odds,
                    blue_odds=blue_odds,
                    current_bankroll=results_dict[f"BANKROLL_{model}"][-1],
                )

                red_wagers, blue_wagers = kelly()
                sliced_df[f"RED_WAGER_{model}"] = red_wagers
                sliced_df[f"BLUE_WAGER_{model}"] = blue_wagers

                total_wager = (
                    sliced_df[f"RED_WAGER_{model}"].sum()
                    + sliced_df[f"BLUE_WAGER_{model}"].sum()
                )
                results_dict[f"TOTAL_WAGER_{model}"].append(
                    round(results_dict[f"TOTAL_WAGER_{model}"][-1] + total_wager, 2)
                )

                sliced_df[f"RED_PROFIT_{model}"] = sliced_df.apply(
                    lambda row: self.helper_calculate_profit(
                        "RED",
                        row["RED_WIN"],
                        row[f"RED_WAGER_{model}"],
                        row["RED_FIGHTER_ODDS"],
                    ),
                    axis=1,
                ).round(2)
                sliced_df[f"BLUE_PROFIT_{model}"] = sliced_df.apply(
                    lambda row: self.helper_calculate_profit(
                        "BLUE",
                        row["RED_WIN"],
                        row[f"BLUE_WAGER_{model}"],
                        row["BLUE_FIGHTER_ODDS"],
                    ),
                    axis=1,
                ).round(2)

                total_profit = (
                    sliced_df[f"RED_PROFIT_{model}"].sum()
                    + sliced_df[f"BLUE_PROFIT_{model}"].sum()
                )
                results_dict[f"BANKROLL_{model}"].append(
                    round(results_dict[f"BANKROLL_{model}"][-1] + total_profit, 2)
                )

                roi = (
                    100.0
                    * (results_dict[f"BANKROLL_{model}"][-1] - self.initial_bankroll)
                    / results_dict[f"TOTAL_WAGER_{model}"][-1]
                    if results_dict[f"TOTAL_WAGER_{model}"][-1] > 0
                    else 0.0
                )
                results_dict[f"ROI_{model}"].append(roi)

        return pd.DataFrame(results_dict)

    def plot_bankrolls_over_time(self, results_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(
            results_df["DATES"], results_df["BANKROLL_LR"], label="Logistic Regression"
        )
        ax.step(results_df["DATES"], results_df["BANKROLL_RF"], label="Random Forest")
        ax.step(
            results_df["DATES"], results_df["BANKROLL_GBM"], label="Gradient Boosting"
        )
        ax.step(results_df["DATES"], results_df["BANKROLL_ENSEMBLE"], label="Ensemble")
        ax.hlines(
            y=self.initial_bankroll,
            xmin=results_df["DATES"].min(),
            xmax=results_df["DATES"].max(),
            color="grey",
            linestyle="--",
        )
        ax.hlines(
            y=0,
            xmin=results_df["DATES"].min(),
            xmax=results_df["DATES"].max(),
            color="red",
            linestyle="--",
        )
        ax.set_title("Bankroll Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Bankroll ($)")
        ax.legend()
        plt.show()

    def plot_roi_over_time(self, results_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(results_df["DATES"], results_df["ROI_LR"], label="Logistic Regression")
        ax.step(results_df["DATES"], results_df["ROI_RF"], label="Random Forest")
        ax.step(results_df["DATES"], results_df["ROI_GBM"], label="Gradient Boosting")
        ax.step(results_df["DATES"], results_df["ROI_ENSEMBLE"], label="Ensemble")
        ax.hlines(
            y=0,
            xmin=results_df["DATES"].min(),
            xmax=results_df["DATES"].max(),
            color="grey",
            linestyle="--",
        )
        ax.set_title("ROI Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("ROI (%)")
        ax.legend()
        plt.show()

    def __call__(self) -> None:
        model_probs = self.get_model_probs()
        results_df = self.run_backtest(model_probs)
        self.plot_bankrolls_over_time(results_df)
        self.plot_roi_over_time(results_df)

        # Display the final results
        display(
            results_df[
                [
                    "BANKROLL_LR",
                    "BANKROLL_RF",
                    "BANKROLL_GBM",
                    "BANKROLL_ENSEMBLE",
                    "ROI_LR",
                    "ROI_RF",
                    "ROI_GBM",
                    "ROI_ENSEMBLE",
                ]
            ].tail(1)
        )
