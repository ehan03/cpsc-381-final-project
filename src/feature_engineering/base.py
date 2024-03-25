# standard library imports
import os
import sqlite3
from functools import reduce
from typing import List, Tuple

# third party imports
import pandas as pd

# local imports


class BaseFeatureGenerator:
    """
    Base class for creating features from data
    """

    TRAIN_CUTOFF_DATE = "2010-01-01"
    TRAIN_TEST_SPLIT_DATE = "2022-01-01"

    UFCSTATS_DB = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "ufcstats.db"
    )
    FIGHTMATRIX_DB = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "fightmatrix.db"
    )
    FIGHTODDSIO_DB = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "fightoddsio.db"
    )
    SHERDOG_DB = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "sherdog.db"
    )

    def __init__(self) -> None:
        self.conn = sqlite3.connect(self.UFCSTATS_DB)
        self.conn.execute("ATTACH DATABASE ? AS fightmatrix", (self.FIGHTMATRIX_DB,))
        self.conn.execute("ATTACH DATABASE ? AS fightoddsio", (self.FIGHTODDSIO_DB,))
        self.conn.execute("ATTACH DATABASE ? AS sherdog", (self.SHERDOG_DB,))

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
        ].drop(columns=["DATE"])
        test_df = final_df.loc[final_df["DATE"] >= self.TRAIN_TEST_SPLIT_DATE].drop(
            columns=["DATE"]
        )

        return train_df, test_df
