# standard library imports
import os
import sqlite3

# third party imports

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
