# standard library imports
from typing import Tuple

# third party imports
import pandas as pd

# local imports
from .base import BaseFeatureGenerator


class FightMatrixFeatureGenerator(BaseFeatureGenerator):
    """
    Class for creating features from FightMatrix data
    """

    def create_elo_features(self) -> pd.DataFrame:
        elo_features = pd.read_sql(
            """
            WITH stacked_elo_raw AS (
              SELECT 
                *, 
                ROW_NUMBER() OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    DATE, 
                    EVENT_ID, 
                    BOUT_ORDINAL
                ) AS FIGHTER_BOUT_NUMBER 
              FROM 
                (
                  SELECT 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    FIGHTER_1_ID AS FIGHTER_ID, 
                    FIGHTER_1_ELO_MODIFIED_PRE AS FIGHTER_ELO_MODIFIED_PRE, 
                    FIGHTER_1_ELO_MODIFIED_POST - FIGHTER_1_ELO_MODIFIED_PRE AS FIGHTER_ELO_MODIFIED_CHANGE, 
                    FIGHTER_1_GLICKO1_PRE AS FIGHTER_GLICKO1_PRE, 
                    FIGHTER_1_GLICKO1_POST - FIGHTER_1_GLICKO1_PRE AS FIGHTER_GLICKO1_CHANGE, 
                    FIGHTER_2_ELO_MODIFIED_PRE AS OPPONENT_ELO_MODIFIED_PRE, 
                    FIGHTER_2_ELO_MODIFIED_POST - FIGHTER_2_ELO_MODIFIED_PRE AS OPPONENT_ELO_MODIFIED_CHANGE, 
                    FIGHTER_2_GLICKO1_PRE AS OPPONENT_GLICKO1_PRE, 
                    FIGHTER_2_GLICKO1_POST - FIGHTER_2_GLICKO1_PRE AS OPPONENT_GLICKO1_CHANGE 
                  FROM 
                    fightmatrix.FIGHTMATRIX_BOUTS 
                  UNION ALL 
                  SELECT 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    FIGHTER_2_ID AS FIGHTER_ID, 
                    FIGHTER_2_ELO_MODIFIED_PRE AS FIGHTER_ELO_MODIFIED_PRE, 
                    FIGHTER_2_ELO_MODIFIED_POST - FIGHTER_2_ELO_MODIFIED_PRE AS FIGHTER_ELO_MODIFIED_CHANGE, 
                    FIGHTER_2_GLICKO1_PRE AS FIGHTER_GLICKO1_PRE, 
                    FIGHTER_2_GLICKO1_POST - FIGHTER_2_GLICKO1_PRE AS FIGHTER_GLICKO1_CHANGE, 
                    FIGHTER_1_ELO_MODIFIED_PRE AS OPPONENT_ELO_MODIFIED_PRE, 
                    FIGHTER_1_ELO_MODIFIED_POST - FIGHTER_1_ELO_MODIFIED_PRE AS OPPONENT_ELO_MODIFIED_CHANGE, 
                    FIGHTER_1_GLICKO1_PRE AS OPPONENT_GLICKO1_PRE, 
                    FIGHTER_1_GLICKO1_POST - FIGHTER_1_GLICKO1_PRE AS OPPONENT_GLICKO1_CHANGE 
                  FROM 
                    fightmatrix.FIGHTMATRIX_BOUTS
                ) 
            ), 
            stacked_elo_features AS (
              SELECT 
                EVENT_ID, 
                DATE, 
                BOUT_ORDINAL, 
                FIGHTER_BOUT_NUMBER, 
                FIGHTER_ID, 
                FIGHTER_ELO_MODIFIED_PRE, 
                AVG(FIGHTER_ELO_MODIFIED_PRE) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    FIGHTER_BOUT_NUMBER ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS FIGHTER_ELO_MODIFIED_AVERAGE, 
                LAG(FIGHTER_ELO_MODIFIED_CHANGE, 1) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    FIGHTER_BOUT_NUMBER
                ) AS FIGHTER_ELO_MODIFIED_CHANGE_PREV, 
                AVG(FIGHTER_ELO_MODIFIED_CHANGE) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    FIGHTER_BOUT_NUMBER ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS FIGHTER_ELO_MODIFIED_CHANGE_AVERAGE, 
                AVG(OPPONENT_ELO_MODIFIED_PRE) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    FIGHTER_BOUT_NUMBER ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS OPPONENT_ELO_MODIFIED_AVERAGE, 
                FIGHTER_GLICKO1_PRE, 
                AVG(FIGHTER_GLICKO1_PRE) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    FIGHTER_BOUT_NUMBER ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS FIGHTER_GLICKO1_AVERAGE, 
                LAG(FIGHTER_GLICKO1_CHANGE, 1) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    FIGHTER_BOUT_NUMBER
                ) AS FIGHTER_GLICKO1_CHANGE_PREV, 
                AVG(FIGHTER_GLICKO1_CHANGE) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    FIGHTER_BOUT_NUMBER ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS FIGHTER_GLICKO1_CHANGE_AVERAGE
              FROM 
                stacked_elo_raw 
            ), 
            bout_num_by_fighter AS (
              SELECT 
                *, 
                ROW_NUMBER() OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    DATE, 
                    EVENT_ID, 
                    BOUT_ORDINAL
                ) AS FIGHTER_BOUT_NUMBER 
              FROM 
                (
                  SELECT 
                    BOUT_ID, 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    RED_FIGHTER_ID AS FIGHTER_ID 
                  FROM 
                    main.UFCSTATS_BOUTS_OVERALL 
                  UNION ALL 
                  SELECT 
                    BOUT_ID, 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    BLUE_FIGHTER_ID AS FIGHTER_ID 
                  FROM 
                    main.UFCSTATS_BOUTS_OVERALL
                ) 
            ), 
            ufcstats_fightmatrix_merged_temp AS (
              SELECT 
                t1.BOUT_ID, 
                t1.FIGHTER_ID, 
                t1.FIGHTER_BOUT_NUMBER, 
                t3.FIGHTER_ELO_MODIFIED_PRE, 
                t3.FIGHTER_ELO_MODIFIED_AVERAGE, 
                t3.FIGHTER_ELO_MODIFIED_CHANGE_PREV, 
                t3.FIGHTER_ELO_MODIFIED_CHANGE_AVERAGE, 
                t3.OPPONENT_ELO_MODIFIED_AVERAGE, 
                t3.FIGHTER_GLICKO1_PRE, 
                t3.FIGHTER_GLICKO1_AVERAGE, 
                t3.FIGHTER_GLICKO1_CHANGE_PREV, 
                t3.FIGHTER_GLICKO1_CHANGE_AVERAGE
              FROM 
                bout_num_by_fighter AS t1 
                INNER JOIN fightmatrix.FIGHTMATRIX_FIGHTER_LINKAGE AS t2 ON t1.FIGHTER_ID = t2.UFCSTATS_FIGHTER_ID 
                INNER JOIN stacked_elo_features AS t3 ON t2.FIGHTMATRIX_FIGHTER_ID = t3.FIGHTER_ID 
                AND t1.FIGHTER_BOUT_NUMBER = t3.FIGHTER_BOUT_NUMBER
            ) 
            SELECT 
              t1.BOUT_ID, 
              t1.EVENT_ID, 
              t1.DATE, 
              t1.BOUT_ORDINAL, 
              t1.RED_FIGHTER_ID, 
              t1.BLUE_FIGHTER_ID, 
              t2.FIGHTER_ELO_MODIFIED_PRE - t3.FIGHTER_ELO_MODIFIED_PRE AS FIGHTMATRIX_ELO_DIFF, 
              CAST(
                t2.FIGHTER_ELO_MODIFIED_PRE AS FLOAT
              ) / t3.FIGHTER_ELO_MODIFIED_PRE AS FIGHTMATRIX_ELO_RATIO, 
              t2.FIGHTER_ELO_MODIFIED_AVERAGE - t3.FIGHTER_ELO_MODIFIED_AVERAGE AS FIGHTMATRIX_ELO_AVERAGE_DIFF, 
              CAST(
                t2.FIGHTER_ELO_MODIFIED_AVERAGE AS FLOAT
              ) / t3.FIGHTER_ELO_MODIFIED_AVERAGE AS FIGHTMATRIX_ELO_AVERAGE_RATIO, 
              t2.FIGHTER_ELO_MODIFIED_CHANGE_PREV - t3.FIGHTER_ELO_MODIFIED_CHANGE_PREV AS FIGHTMATRIX_ELO_CHANGE_PREV_DIFF, 
              t2.FIGHTER_ELO_MODIFIED_CHANGE_AVERAGE - t3.FIGHTER_ELO_MODIFIED_CHANGE_AVERAGE AS FIGHTMATRIX_ELO_CHANGE_AVERAGE_DIFF, 
              t2.OPPONENT_ELO_MODIFIED_AVERAGE - t3.OPPONENT_ELO_MODIFIED_AVERAGE AS FIGHTMATRIX_OPPONENT_ELO_AVERAGE_DIFF, 
              CAST(
                t2.OPPONENT_ELO_MODIFIED_AVERAGE AS FLOAT
              ) / t3.OPPONENT_ELO_MODIFIED_AVERAGE AS FIGHTMATRIX_OPPONENT_ELO_AVERAGE_RATIO, 
              t2.FIGHTER_GLICKO1_PRE - t3.FIGHTER_GLICKO1_PRE AS FIGHTMATRIX_GLICKO1_DIFF, 
              CAST(
                t2.FIGHTER_GLICKO1_AVERAGE AS FLOAT
              ) / t3.FIGHTER_GLICKO1_AVERAGE AS FIGHTMATRIX_GLICKO1_AVERAGE_RATIO, 
              t2.FIGHTER_GLICKO1_CHANGE_PREV - t3.FIGHTER_GLICKO1_CHANGE_PREV AS FIGHTMATRIX_GLICKO1_CHANGE_PREV_DIFF, 
              t2.FIGHTER_GLICKO1_CHANGE_AVERAGE - t3.FIGHTER_GLICKO1_CHANGE_AVERAGE AS FIGHTMATRIX_GLICKO1_CHANGE_AVERAGE_DIFF, 
              CASE t1.RED_OUTCOME WHEN 'W' THEN 1 WHEN 'L' THEN 0 ELSE NULL END AS RED_WIN 
            FROM 
              main.UFCSTATS_BOUTS_OVERALL AS t1 
              INNER JOIN ufcstats_fightmatrix_merged_temp AS t2 ON t1.BOUT_ID = t2.BOUT_ID 
              AND t1.RED_FIGHTER_ID = t2.FIGHTER_ID 
              INNER JOIN ufcstats_fightmatrix_merged_temp AS t3 ON t1.BOUT_ID = t3.BOUT_ID 
              AND t1.BLUE_FIGHTER_ID = t3.FIGHTER_ID 
            WHERE 
              t2.FIGHTER_BOUT_NUMBER > 1 
              AND t3.FIGHTER_BOUT_NUMBER > 1 
              AND t1.DATE >= ? 
            ORDER BY 
              t1.DATE, 
              t1.EVENT_ID, 
              t1.BOUT_ORDINAL;
            """,
            self.conn,
            params=[self.TRAIN_CUTOFF_DATE],
        ).drop(
            columns=["EVENT_ID", "BOUT_ORDINAL", "RED_FIGHTER_ID", "BLUE_FIGHTER_ID"]
        )

        return elo_features

    def create_ranking_features(self) -> pd.DataFrame:
        ranking_features = pd.read_sql(
            """
            WITH bout_num_by_fighter AS (
              SELECT 
                *, 
                ROW_NUMBER() OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    DATE, 
                    EVENT_ID, 
                    BOUT_ORDINAL
                ) AS FIGHTER_BOUT_NUMBER 
              FROM 
                (
                  SELECT 
                    BOUT_ID, 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    RED_FIGHTER_ID AS FIGHTER_ID 
                  FROM 
                    main.UFCSTATS_BOUTS_OVERALL 
                  UNION ALL 
                  SELECT 
                    BOUT_ID, 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    BLUE_FIGHTER_ID AS FIGHTER_ID 
                  FROM 
                    main.UFCSTATS_BOUTS_OVERALL
                ) 
              ORDER BY 
                DATE, 
                EVENT_ID, 
                BOUT_ORDINAL
            ), 
            stacked_ranking_raw AS (
              SELECT
                BOUT_ID,
                FIGHTER_ID,
                FIGHTER_BOUT_NUMBER,
                CASE 
                  WHEN POINTS IS NULL AND FIGHTER_BOUT_NUMBER > 1 THEN 20
                  ELSE POINTS
                END AS POINTS
              FROM (
                SELECT
                  t1.*,
                  t3.RANK,
                  t3.POINTS,
                  ROW_NUMBER() OVER (PARTITION BY t1.FIGHTER_ID, t1.BOUT_ID ORDER BY t3.ISSUE_DATE DESC) AS rn
                FROM
                  bout_num_by_fighter AS t1
                INNER JOIN
                  fightmatrix.FIGHTMATRIX_FIGHTER_LINKAGE AS t2
                ON
                  t1.FIGHTER_ID = t2.UFCSTATS_FIGHTER_ID
                LEFT JOIN
                  fightmatrix.FIGHTMATRIX_RANKINGS AS t3
                ON
                  t2.FIGHTMATRIX_FIGHTER_ID = t3.FIGHTER_ID
                  AND t1.DATE > t3.ISSUE_DATE
              )
              WHERE
                rn = 1
            ),
            stacked_ranking_features AS (
              SELECT
                BOUT_ID,
                FIGHTER_ID,
                FIGHTER_BOUT_NUMBER,
                POINTS AS FIGHTMATRIX_RANKING_POINTS,
                AVG(POINTS) OVER (PARTITION BY FIGHTER_ID ORDER BY FIGHTER_BOUT_NUMBER) AS FIGHTMATRIX_RANKING_POINTS_AVERAGE
              FROM
                stacked_ranking_raw
            )
            SELECT
              t1.BOUT_ID,
              t1.EVENT_ID,
              t1.DATE,
              t1.BOUT_ORDINAL,
              t1.RED_FIGHTER_ID,
              t1.BLUE_FIGHTER_ID,
              t2.FIGHTMATRIX_RANKING_POINTS - t3.FIGHTMATRIX_RANKING_POINTS AS FIGHTMATRIX_RANKING_POINTS_DIFF,
              CAST(t2.FIGHTMATRIX_RANKING_POINTS AS FLOAT) / t3.FIGHTMATRIX_RANKING_POINTS AS FIGHTMATRIX_RANKING_POINTS_RATIO,
              t2.FIGHTMATRIX_RANKING_POINTS_AVERAGE - t3.FIGHTMATRIX_RANKING_POINTS_AVERAGE AS FIGHTMATRIX_RANKING_POINTS_AVERAGE_DIFF,
              CAST(t2.FIGHTMATRIX_RANKING_POINTS_AVERAGE AS FLOAT) / t3.FIGHTMATRIX_RANKING_POINTS_AVERAGE AS FIGHTMATRIX_RANKING_POINTS_AVERAGE_RATIO,
              CASE t1.RED_OUTCOME WHEN 'W' THEN 1 WHEN 'L' THEN 0 ELSE NULL END AS RED_WIN 
            FROM
              main.UFCSTATS_BOUTS_OVERALL AS t1
            INNER JOIN
              stacked_ranking_features AS t2
            ON
              t1.BOUT_ID = t2.BOUT_ID
              AND t1.RED_FIGHTER_ID = t2.FIGHTER_ID
            INNER JOIN
              stacked_ranking_features AS t3
            ON
              t1.BOUT_ID = t3.BOUT_ID
              AND t1.BLUE_FIGHTER_ID = t3.FIGHTER_ID
            WHERE
              t2.FIGHTER_BOUT_NUMBER > 1
              AND t3.FIGHTER_BOUT_NUMBER > 1
              AND t1.DATE >= ?
            ORDER BY
              t1.DATE,
              t1.EVENT_ID,
              t1.BOUT_ORDINAL;
            """,
            self.conn,
            params=[self.TRAIN_CUTOFF_DATE],
        ).drop(
            columns=["EVENT_ID", "BOUT_ORDINAL", "RED_FIGHTER_ID", "BLUE_FIGHTER_ID"]
        )

        return ranking_features

    def __call__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        elo_features = self.create_elo_features()
        ranking_features = self.create_ranking_features()
        train_df, test_df = self.create_train_test_dfs([elo_features, ranking_features])

        return train_df, test_df
