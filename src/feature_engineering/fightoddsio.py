# standard library imports
from typing import Tuple

# third party imports
import pandas as pd

# local imports
from .base import BaseFeatureGenerator


class FightOddsIOFeatureGenerator(BaseFeatureGenerator):
    """
    Class for generating features from FightOdds.io data
    """

    def create_odds_features(self) -> pd.DataFrame:
        odds_features = pd.read_sql(
            """
            WITH fightoddsio_bouts_full_odds AS (
              SELECT 
                EVENT_SLUG, 
                DATE, 
                FIGHTER_1_ID, 
                FIGHTER_2_ID, 
                FIGHTER_1_ODDS, 
                FIGHTER_2_ODDS 
              FROM 
                fightoddsio.FIGHTODDSIO_BOUTS 
              WHERE 
                FIGHTER_1_ODDS IS NOT NULL 
                AND FIGHTER_2_ODDS IS NOT NULL
            ), 
            odds_merge_temp AS (
              SELECT 
                t1.BOUT_ID, 
                t1.EVENT_ID, 
                t1.DATE, 
                t1.BOUT_ORDINAL, 
                t1.RED_FIGHTER_ID, 
                t1.BLUE_FIGHTER_ID, 
                t1.RED_OUTCOME, 
                t1.BLUE_OUTCOME, 
                t2.RED_FIGHTER_ODDS, 
                t2.BLUE_FIGHTER_ODDS, 
                1 / t2.RED_FIGHTER_ODDS AS RED_FIGHTER_PROB_VIG, 
                1 / t2.BLUE_FIGHTER_ODDS AS BLUE_FIGHTER_PROB_VIG 
              FROM 
                main.UFCSTATS_BOUTS_OVERALL AS t1 
                INNER JOIN main.BESTFIGHTODDS_HISTORICAL_ODDS AS t2 ON t1.BOUT_ID = t2.UFCSTATS_BOUT_ID 
              UNION 
              SELECT 
                t1.BOUT_ID, 
                t1.EVENT_ID, 
                t1.DATE, 
                t1.BOUT_ORDINAL, 
                t1.RED_FIGHTER_ID, 
                t1.BLUE_FIGHTER_ID, 
                t1.RED_OUTCOME, 
                t1.BLUE_OUTCOME, 
                t5.FIGHTER_1_ODDS AS RED_FIGHTER_ODDS, 
                t5.FIGHTER_2_ODDS AS BLUE_FIGHTER_ODDS, 
                1 / t5.FIGHTER_1_ODDS AS RED_FIGHTER_PROB_VIG, 
                1 / t5.FIGHTER_2_ODDS AS BLUE_FIGHTER_PROB_VIG 
              FROM 
                main.UFCSTATS_BOUTS_OVERALL AS t1 
                INNER JOIN fightoddsio.FIGHTODDSIO_EVENT_LINKAGE AS t2 ON t1.EVENT_ID = t2.UFCSTATS_EVENT_ID 
                INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t3 ON t1.RED_FIGHTER_ID = t3.UFCSTATS_FIGHTER_ID 
                INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t4 ON t1.BLUE_FIGHTER_ID = t4.UFCSTATS_FIGHTER_ID 
                INNER JOIN fightoddsio_bouts_full_odds AS t5 ON t2.FIGHTODDSIO_EVENT_SLUG = t5.EVENT_SLUG 
                AND t3.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_1_ID 
                AND t4.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_2_ID 
              UNION 
              SELECT 
                t1.BOUT_ID, 
                t1.EVENT_ID, 
                t1.DATE, 
                t1.BOUT_ORDINAL, 
                t1.RED_FIGHTER_ID, 
                t1.BLUE_FIGHTER_ID, 
                t1.RED_OUTCOME, 
                t1.BLUE_OUTCOME, 
                t5.FIGHTER_2_ODDS AS RED_FIGHTER_ODDS, 
                t5.FIGHTER_1_ODDS AS BLUE_FIGHTER_ODDS, 
                1 / t5.FIGHTER_2_ODDS AS RED_FIGHTER_PROB_VIG, 
                1 / t5.FIGHTER_1_ODDS AS BLUE_FIGHTER_PROB_VIG 
              FROM 
                main.UFCSTATS_BOUTS_OVERALL AS t1 
                INNER JOIN fightoddsio.FIGHTODDSIO_EVENT_LINKAGE AS t2 ON t1.EVENT_ID = t2.UFCSTATS_EVENT_ID 
                INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t3 ON t1.RED_FIGHTER_ID = t3.UFCSTATS_FIGHTER_ID 
                INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t4 ON t1.BLUE_FIGHTER_ID = t4.UFCSTATS_FIGHTER_ID 
                INNER JOIN fightoddsio_bouts_full_odds AS t5 ON t2.FIGHTODDSIO_EVENT_SLUG = t5.EVENT_SLUG 
                AND t3.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_2_ID 
                AND t4.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_1_ID
            ), 
            stacked_odds_raw AS (
              SELECT 
                *, 
                ROW_NUMBER() OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    DATE, 
                    EVENT_ID, 
                    BOUT_ORDINAL
                ) -1 AS rn 
              FROM 
                (
                  SELECT 
                    BOUT_ID, 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    RED_FIGHTER_ID AS FIGHTER_ID, 
                    RED_FIGHTER_ODDS AS FIGHTER_ODDS, 
                    BLUE_FIGHTER_ODDS AS OPPONENT_ODDS, 
                    RED_FIGHTER_PROB_VIG / (
                      RED_FIGHTER_PROB_VIG + BLUE_FIGHTER_PROB_VIG
                    ) AS FIGHTER_IMPLIED_PROB, 
                    CASE WHEN RED_OUTCOME = 'W' 
                    AND RED_FIGHTER_ODDS <= BLUE_FIGHTER_ODDS THEN 1 ELSE 0 END AS WIN_FAVORITE, 
                    CASE WHEN RED_OUTCOME = 'W' 
                    AND RED_FIGHTER_ODDS > BLUE_FIGHTER_ODDS THEN 1 ELSE 0 END AS WIN_UNDERDOG, 
                    CASE WHEN RED_OUTCOME = 'L' 
                    AND RED_FIGHTER_ODDS <= BLUE_FIGHTER_ODDS THEN 1 ELSE 0 END AS LOSE_FAVORITE, 
                    CASE WHEN RED_OUTCOME = 'L' 
                    AND RED_FIGHTER_ODDS > BLUE_FIGHTER_ODDS THEN 1 ELSE 0 END AS LOSE_UNDERDOG, 
                    CASE WHEN RED_OUTCOME = 'W' THEN RED_FIGHTER_ODDS - 1 WHEN RED_OUTCOME = 'L' THEN -1 ELSE 0 END AS ROI 
                  FROM 
                    odds_merge_temp 
                  UNION ALL 
                  SELECT 
                    BOUT_ID, 
                    EVENT_ID, 
                    DATE, 
                    BOUT_ORDINAL, 
                    BLUE_FIGHTER_ID AS FIGHTER_ID, 
                    BLUE_FIGHTER_ODDS AS FIGHTER_ODDS, 
                    RED_FIGHTER_ODDS AS OPPONENT_ODDS, 
                    BLUE_FIGHTER_PROB_VIG / (
                      RED_FIGHTER_PROB_VIG + BLUE_FIGHTER_PROB_VIG
                    ) AS FIGHTER_IMPLIED_PROB, 
                    CASE WHEN BLUE_OUTCOME = 'W' 
                    AND BLUE_FIGHTER_ODDS < RED_FIGHTER_ODDS THEN 1 ELSE 0 END AS WIN_FAVORITE, 
                    CASE WHEN BLUE_OUTCOME = 'W' 
                    AND BLUE_FIGHTER_ODDS > RED_FIGHTER_ODDS THEN 1 ELSE 0 END AS WIN_UNDERDOG, 
                    CASE WHEN BLUE_OUTCOME = 'L' 
                    AND BLUE_FIGHTER_ODDS < RED_FIGHTER_ODDS THEN 1 ELSE 0 END AS LOSE_FAVORITE, 
                    CASE WHEN BLUE_OUTCOME = 'L' 
                    AND BLUE_FIGHTER_ODDS > RED_FIGHTER_ODDS THEN 1 ELSE 0 END AS LOSE_UNDERDOG, 
                    CASE WHEN BLUE_OUTCOME = 'W' THEN BLUE_FIGHTER_ODDS - 1 WHEN BLUE_OUTCOME = 'L' THEN -1 ELSE 0 END AS ROI 
                  FROM 
                    odds_merge_temp
                )
            ), 
            stacked_odds_features AS (
              SELECT 
                BOUT_ID, 
                FIGHTER_ID, 
                AVG(FIGHTER_ODDS) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS FIGHTER_ODDS_AVERAGE, 
                AVG(OPPONENT_ODDS) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS OPPONENT_ODDS_AVERAGE, 
                AVG(FIGHTER_IMPLIED_PROB) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS FIGHTER_IMPLIED_PROB_AVERAGE, 
                SUM(WIN_FAVORITE) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) / CAST(rn AS FLOAT) AS WIN_FAVORITE_RATE, 
                SUM(WIN_UNDERDOG) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) / CAST(rn AS FLOAT) AS WIN_UNDERDOG_RATE, 
                SUM(LOSE_FAVORITE) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) / CAST(rn AS FLOAT) AS LOSE_FAVORITE_RATE, 
                SUM(LOSE_UNDERDOG) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) / CAST(rn AS FLOAT) AS LOSE_UNDERDOG_RATE, 
                AVG(ROI) OVER(
                  PARTITION BY FIGHTER_ID 
                  ORDER BY 
                    rn ROWS BETWEEN UNBOUNDED PRECEDING 
                    AND 1 PRECEDING
                ) AS ROI_AVERAGE 
              FROM 
                stacked_odds_raw
            ), 
            bout_num_by_fighter AS (
              SELECT 
                BOUT_ID, 
                FIGHTER_ID, 
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
            ) 
            SELECT 
              t1.BOUT_ID, 
              t1.EVENT_ID, 
              t1.DATE, 
              t1.BOUT_ORDINAL, 
              t1.RED_FIGHTER_ID, 
              t1.BLUE_FIGHTER_ID, 
              t2.FIGHTER_ODDS_AVERAGE - t3.FIGHTER_ODDS_AVERAGE AS ODDS_AVERAGE_DIFF, 
              t2.OPPONENT_ODDS_AVERAGE - t3.OPPONENT_ODDS_AVERAGE AS OPPONENT_ODDS_AVERAGE_DIFF, 
              t2.FIGHTER_IMPLIED_PROB_AVERAGE - t3.FIGHTER_IMPLIED_PROB_AVERAGE AS IMPLIED_PROB_AVERAGE_DIFF, 
              t2.WIN_FAVORITE_RATE - t3.WIN_FAVORITE_RATE AS WIN_FAVORITE_RATE_DIFF, 
              t2.WIN_UNDERDOG_RATE - t3.WIN_UNDERDOG_RATE AS WIN_UNDERDOG_RATE_DIFF, 
              t2.LOSE_FAVORITE_RATE - t3.LOSE_FAVORITE_RATE AS LOSE_FAVORITE_RATE_DIFF, 
              t2.LOSE_UNDERDOG_RATE - t3.LOSE_UNDERDOG_RATE AS LOSE_UNDERDOG_RATE_DIFF, 
              t2.ROI_AVERAGE - t3.ROI_AVERAGE AS ROI_AVERAGE_DIFF, 
              CASE t1.RED_OUTCOME WHEN 'W' THEN 1 WHEN 'L' THEN 0 ELSE NULL END AS RED_WIN 
            FROM 
              main.UFCSTATS_BOUTS_OVERALL AS t1 
              LEFT JOIN stacked_odds_features AS t2 ON t1.BOUT_ID = t2.BOUT_ID 
              AND t1.RED_FIGHTER_ID = t2.FIGHTER_ID 
              LEFT JOIN stacked_odds_features AS t3 ON t1.BOUT_ID = t3.BOUT_ID 
              AND t1.BLUE_FIGHTER_ID = t3.FIGHTER_ID 
              LEFT JOIN bout_num_by_fighter AS t4 ON t1.BOUT_ID = t4.BOUT_ID 
              AND t1.RED_FIGHTER_ID = t4.FIGHTER_ID 
              LEFT JOIN bout_num_by_fighter AS t5 ON t1.BOUT_ID = t5.BOUT_ID 
              AND t1.BLUE_FIGHTER_ID = t5.FIGHTER_ID 
            WHERE 
              t4.FIGHTER_BOUT_NUMBER > 1 
              AND t5.FIGHTER_BOUT_NUMBER > 1 
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

        return odds_features

    def __call__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        odds_features = self.create_odds_features()
        train_df, test_df = self.create_train_test_dfs([odds_features])

        return train_df, test_df
