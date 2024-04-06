# standard library imports
import os

# third-party imports
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
        """
        Initialize the ModelTrainer class
        """

        if model_name not in [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        ]:
            raise ValueError(
                """Invalid model name. Please choose one of 'logistic_regression', 
                'random_forest', or 'gradient_boosting'."""
            )

        self.model_name = model_name
        self.train_df = pd.read_csv(
            os.path.join(self.DATA_PATH, "processed", "train.csv")
        ).drop(columns=["BOUT_ID", "EVENT_ID", "DATE", "BOUT_ORDINAL"])

    def train_model(self) -> object:
        """
        Fit the specified model to the training data and
        return the estimator/pipeline
        """

        X_train = self.train_df.drop(columns=["RED_WIN"])
        y_train = self.train_df["RED_WIN"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = None

        if self.model_name == "logistic_regression":
            pipe = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classify",
                        LogisticRegression(penalty="l2", max_iter=300, random_state=0),
                    ),
                ]
            )

            param_grid = {
                "classify__C": np.logspace(-4, 4, 10),
            }
            grid_search = GridSearchCV(
                pipe, param_grid, cv=cv, scoring="neg_log_loss", n_jobs=-1, refit=True
            )
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            model = grid_search

        elif self.model_name == "random_forest":
            param_grid = {
                "max_depth": [7, 12, 15],
                "max_leaf_nodes": [None, 70, 80],
            }
            clf = RandomForestClassifier(
                n_estimators=200,
                criterion="gini",
                max_features="sqrt",
                random_state=0,
                n_jobs=-1,
            )
            grid_search = GridSearchCV(
                clf, param_grid, cv=cv, scoring="neg_log_loss", n_jobs=-1, refit=True
            )
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            model = grid_search

        elif self.model_name == "gradient_boosting":
            param_grid = {
                "learning_rate": [0.01, 0.1],
                "max_depth": [2, 3, 5],
            }
            clf = GradientBoostingClassifier(n_estimators=100, random_state=0, max_features="sqrt", max_leaf_nodes=None)
            grid_search = GridSearchCV(
                clf, param_grid, cv=cv, scoring="neg_log_loss", n_jobs=-1, refit=True
            )
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            model = grid_search

        return model

    def __call__(self) -> None:
        """
        Train the specified model and save it to disk
        """

        model = self.train_model()
        model_path = os.path.join(self.MODELS_PATH, f"{self.model_name}.joblib")
        dump(model, model_path)
