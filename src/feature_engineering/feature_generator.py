# standard library imports
import math
import os
import sqlite3
from functools import reduce
from typing import List, Tuple

# third party imports
import pandas as pd

# local imports


class FeatureGenerator:
    """
    Class for creating features from data
    """

    # Split/cutoff dates
    TRAIN_CUTOFF_DATE = "2010-01-01"
    TRAIN_TEST_SPLIT_DATE = "2022-01-01"

    # Base paths
    DIR_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(DIR_PATH, "..", "..", "data")
    QUERIES_PATH = os.path.join(DIR_PATH, "queries")

    # SQLite database paths
    UFCSTATS_DB_PATH = os.path.join(DATA_PATH, "ufcstats.db")
    FIGHTMATRIX_DB_PATH = os.path.join(DATA_PATH, "fightmatrix.db")
    FIGHTODDSIO_DB_PATH = os.path.join(DATA_PATH, "fightoddsio.db")
    SHERDOG_DB_PATH = os.path.join(DATA_PATH, "sherdog.db")

    def __init__(self) -> None:
        self.conn = sqlite3.connect(self.UFCSTATS_DB_PATH)
        self.conn.execute(
            "ATTACH DATABASE ? AS fightmatrix", (self.FIGHTMATRIX_DB_PATH,)
        )
        self.conn.execute(
            "ATTACH DATABASE ? AS fightoddsio", (self.FIGHTODDSIO_DB_PATH,)
        )
        self.conn.execute("ATTACH DATABASE ? AS sherdog", (self.SHERDOG_DB_PATH,))
        self.conn.create_function("SQRT", 1, math.sqrt)

    def create_feature_dfs(self) -> List[pd.DataFrame]:
        feature_dfs = []

        for query_file in os.listdir(self.QUERIES_PATH):
            if query_file.endswith("_features.sql"):
                with open(os.path.join(self.QUERIES_PATH, query_file), "r") as f:
                    queries = f.read().split(";")[
                        :-1
                    ]  # Skip last element, empty string
                    for query in queries:
                        df = pd.read_sql_query(
                            query,
                            self.conn,
                            params=[self.TRAIN_CUTOFF_DATE],
                        )
                        feature_dfs.append(df)

        return feature_dfs

    def create_train_test_dfs(
        self, df_list: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        final_df = df_list[0]

        if len(df_list) > 1:
            final_df = reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    on=["BOUT_ID", "EVENT_ID", "DATE", "BOUT_ORDINAL", "RED_WIN"],
                    how="inner",
                ),
                df_list,
            )

        train_df = final_df.loc[
            (final_df["DATE"] >= self.TRAIN_CUTOFF_DATE)
            & (final_df["DATE"] < self.TRAIN_TEST_SPLIT_DATE)
            & (final_df["RED_WIN"].notnull())
        ]
        test_df = final_df.loc[final_df["DATE"] >= self.TRAIN_TEST_SPLIT_DATE]

        return train_df, test_df

    def create_backtest_odds_df(self) -> pd.DataFrame:
        with open(os.path.join(self.QUERIES_PATH, "backtest_odds.sql"), "r") as f:
            query = f.read()
            backtest_odds_df = pd.read_sql(
                query, self.conn, params=[self.TRAIN_CUTOFF_DATE]
            )

        return backtest_odds_df

    def __call__(self) -> None:
        feature_dfs = self.create_feature_dfs()
        train_df, test_df = self.create_train_test_dfs(feature_dfs)
        backtest_odds_df = self.create_backtest_odds_df()

        self.conn.close()

        train_df.to_csv(
            os.path.join(self.DATA_PATH, "processed", "train.csv"), index=False
        )
        test_df.to_csv(
            os.path.join(self.DATA_PATH, "processed", "test.csv"), index=False
        )
        backtest_odds_df.to_csv(
            os.path.join(self.DATA_PATH, "processed", "backtest_odds.csv"), index=False
        )
