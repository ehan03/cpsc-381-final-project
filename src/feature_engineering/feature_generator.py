# standard library imports
import os
import sqlite3
from functools import reduce
from typing import List, Tuple

# third party imports
import pandas as pd

# local imports
from .misc_query import NUM_BOUTS_BY_FIGHTER_QUERY, SHERDOG_STREAKS_QUERY


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

    def create_feature_dfs(self) -> List[pd.DataFrame]:
        feature_dfs = []

        for query_file in os.listdir(self.QUERIES_PATH):
            with open(os.path.join(self.QUERIES_PATH, query_file), "r") as f:
                queries = f.read().split(";")[:-1]  # Skip last query which is empty
                for query in queries:
                    df = pd.read_sql_query(
                        query,
                        self.conn,
                        params=[self.TRAIN_CUTOFF_DATE],
                    ).drop(
                        columns=[
                            "EVENT_ID",
                            "BOUT_ORDINAL",
                        ]
                    )
                    feature_dfs.append(df)

        # Handle annoying edge case with streaks features due to inefficient query
        sherdog_streak_features = pd.read_sql(SHERDOG_STREAKS_QUERY, self.conn)
        ufcstats_num_bouts_by_fighter = pd.read_sql(
            NUM_BOUTS_BY_FIGHTER_QUERY, self.conn
        )
        ufcstats_streak_merge_temp = pd.merge(
            ufcstats_num_bouts_by_fighter,
            sherdog_streak_features,
            left_on=["FIGHTER_ID", "FIGHTER_BOUT_NUMBER"],
            right_on=["UFCSTATS_FIGHTER_ID", "FIGHTER_BOUT_NUMBER"],
        )[
            [
                "BOUT_ID",
                "FIGHTER_ID",
                "FIGHTER_BOUT_NUMBER",
                "WINNING_STREAK",
                "LOSING_STREAK",
                "WINNING_STREAK_AVERAGE",
                "LOSING_STREAK_AVERAGE",
                "WINNING_STREAK_MAX",
                "LOSING_STREAK_MAX",
            ]
        ]

        ufcstats_bouts = pd.read_sql(
            """
            SELECT
              BOUT_ID,
              EVENT_ID,
              DATE,
              BOUT_ORDINAL,
              RED_FIGHTER_ID,
              BLUE_FIGHTER_ID,
              CASE RED_OUTCOME WHEN 'W' THEN 1 WHEN 'L' THEN 0 ELSE NULL END AS RED_WIN
            FROM
              main.UFCSTATS_BOUTS_OVERALL
            WHERE
              DATE >= ?
            ORDER BY
              DATE,
              EVENT_ID,
              BOUT_ORDINAL;
            """,
            self.conn,
            params=[self.TRAIN_CUTOFF_DATE],
        )

        merged_1 = pd.merge(
            ufcstats_bouts,
            ufcstats_streak_merge_temp,
            how="inner",
            left_on=["BOUT_ID", "RED_FIGHTER_ID"],
            right_on=["BOUT_ID", "FIGHTER_ID"],
        )
        merged_2 = pd.merge(
            merged_1,
            ufcstats_streak_merge_temp,
            how="inner",
            left_on=["BOUT_ID", "BLUE_FIGHTER_ID"],
            right_on=["BOUT_ID", "FIGHTER_ID"],
        )
        merged_2["SHERDOG_WINNING_STREAK_DIFF"] = (
            merged_2["WINNING_STREAK_x"] - merged_2["WINNING_STREAK_y"]
        )
        merged_2["SHERDOG_LOSING_STREAK_DIFF"] = (
            merged_2["LOSING_STREAK_x"] - merged_2["LOSING_STREAK_y"]
        )
        merged_2["SHERDOG_WINNING_STREAK_AVERAGE_DIFF"] = (
            merged_2["WINNING_STREAK_AVERAGE_x"] - merged_2["WINNING_STREAK_AVERAGE_y"]
        )
        merged_2["SHERDOG_LOSING_STREAK_AVERAGE_DIFF"] = (
            merged_2["LOSING_STREAK_AVERAGE_x"] - merged_2["LOSING_STREAK_AVERAGE_y"]
        )
        merged_2["SHERDOG_WINNING_STREAK_MAX_DIFF"] = (
            merged_2["WINNING_STREAK_MAX_x"] - merged_2["WINNING_STREAK_MAX_y"]
        )
        merged_2["SHERDOG_LOSING_STREAK_MAX_DIFF"] = (
            merged_2["LOSING_STREAK_MAX_x"] - merged_2["LOSING_STREAK_MAX_y"]
        )

        streaks_df_final = merged_2.loc[
            (merged_2["FIGHTER_BOUT_NUMBER_x"] > 1)
            & (merged_2["FIGHTER_BOUT_NUMBER_y"] > 1)
        ][
            [
                "BOUT_ID",
                "DATE",
                "RED_WIN",
                "SHERDOG_WINNING_STREAK_DIFF",
                "SHERDOG_LOSING_STREAK_DIFF",
                "SHERDOG_WINNING_STREAK_AVERAGE_DIFF",
                "SHERDOG_LOSING_STREAK_AVERAGE_DIFF",
                "SHERDOG_WINNING_STREAK_MAX_DIFF",
                "SHERDOG_LOSING_STREAK_MAX_DIFF",
            ]
        ]

        feature_dfs.append(streaks_df_final)

        self.conn.close()

        return feature_dfs

    def create_train_test_dfs(
        self, df_list: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        final_df = df_list[0]

        if len(df_list) > 1:
            final_df = reduce(
                lambda left, right: pd.merge(
                    left, right, on=["BOUT_ID", "DATE", "RED_WIN"], how="inner"
                ),
                df_list,
            )

        train_df = final_df.loc[
            (final_df["DATE"] >= self.TRAIN_CUTOFF_DATE)
            & (final_df["DATE"] < self.TRAIN_TEST_SPLIT_DATE)
            & (final_df["RED_WIN"].notnull())
        ].drop(columns=["BOUT_ID", "DATE"])
        test_df = final_df.loc[final_df["DATE"] >= self.TRAIN_TEST_SPLIT_DATE].drop(
            columns=["BOUT_ID", "DATE"]
        )

        return train_df, test_df

    def __call__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_dfs = self.create_feature_dfs()
        train_df, test_df = self.create_train_test_dfs(feature_dfs)

        return train_df, test_df
