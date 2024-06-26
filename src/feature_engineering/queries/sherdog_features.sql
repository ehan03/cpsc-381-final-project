-- Main Sherdog features
WITH mma_debuts_and_dobs AS (
  SELECT 
    t1.*, 
    t2.DATE_OF_BIRTH, 
    t3.UFCSTATS_FIGHTER_ID 
  FROM 
    (
      SELECT 
        FIGHTER_ID, 
        MIN(DATE) AS SHERDOG_DEBUT_DATE 
      FROM 
        sherdog.SHERDOG_BOUT_HISTORY 
      GROUP BY 
        FIGHTER_ID
    ) AS t1 
    INNER JOIN sherdog.SHERDOG_FIGHTERS AS t2 ON t1.FIGHTER_ID = t2.FIGHTER_ID 
    INNER JOIN sherdog.SHERDOG_FIGHTER_LINKAGE AS t3 ON t1.FIGHTER_ID = t3.SHERDOG_FIGHTER_ID
), 
mma_debuts_and_dobs_filled AS (
  SELECT 
    t1.FIGHTER_ID, 
    t1.UFCSTATS_FIGHTER_ID, 
    t1.SHERDOG_DEBUT_DATE, 
    CASE WHEN t1.UFCSTATS_FIGHTER_ID = '2721bb808bb2523c' THEN '1980-05-30' WHEN t1.DATE_OF_BIRTH IS NULL THEN t2.DATE_OF_BIRTH ELSE t1.DATE_OF_BIRTH END AS DATE_OF_BIRTH 
  FROM 
    mma_debuts_and_dobs AS t1 
    INNER JOIN main.UFCSTATS_FIGHTERS AS t2 ON t1.UFCSTATS_FIGHTER_ID = t2.FIGHTER_ID
), 
stacked_sherdog_raw AS (
  SELECT 
    t2.UFCSTATS_FIGHTER_ID, 
    t1.EVENT_ID, 
    t1.FIGHTER_BOUT_ORDINAL, 
    julianday(t1.DATE) - julianday(t2.DATE_OF_BIRTH) AS AGE_DAYS, 
    julianday(t1.DATE) - julianday(t2.SHERDOG_DEBUT_DATE) AS DAYS_SINCE_DEBUT, 
    julianday(t1.DATE) - julianday(
      LAG(t1.DATE, 1) OVER (
        PARTITION BY t1.FIGHTER_ID 
        ORDER BY 
          t1.FIGHTER_BOUT_ORDINAL
      )
    ) AS DAYS_SINCE_LAST_FIGHT, 
    CASE t1.OUTCOME WHEN 'W' THEN 1 ELSE 0 END AS WIN, 
    CASE WHEN t1.OUTCOME = 'L' 
    AND t1.OUTCOME_METHOD IN ('KO', 'TKO') THEN 1 ELSE 0 END AS LOSS_BY_KO_TKO, 
    CASE WHEN t1.OUTCOME = 'L' 
    AND t1.OUTCOME_METHOD IN (
      'Submission', 'Technical Submission'
    ) THEN 1 ELSE 0 END AS LOSS_BY_SUBMISSION, 
    CASE WHEN t1.OUTCOME = 'L' 
    AND t1.OUTCOME_METHOD = 'Decision' THEN 1 ELSE 0 END AS LOSS_BY_DECISION, 
    t1.FIGHTER_BOUT_ORDINAL AS TOTAL_FIGHTS, 
    t1.TOTAL_TIME_SECONDS 
  FROM 
    sherdog.SHERDOG_BOUT_HISTORY AS t1 
    LEFT JOIN mma_debuts_and_dobs_filled AS t2 ON t1.FIGHTER_ID = t2.FIGHTER_ID
), 
stacked_sherdog_features AS (
  SELECT 
    *, 
    ROW_NUMBER() OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        TOTAL_FIGHTS
    ) AS FIGHTER_BOUT_NUMBER 
  FROM 
    (
      SELECT 
        UFCSTATS_FIGHTER_ID, 
        EVENT_ID, 
        AGE_DAYS, 
        AVG(AGE_DAYS) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND 1 PRECEDING
        ) AS AGE_DAYS_AVERAGE, 
        DAYS_SINCE_DEBUT, 
        AVG(DAYS_SINCE_LAST_FIGHT) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND CURRENT ROW
        ) AS DAYS_SINCE_LAST_FIGHT_AVERAGE, 
        SUM(WIN) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND 1 PRECEDING
        ) AS WINS, 
        AVG(WIN) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND 1 PRECEDING
        ) AS WIN_RATE, 
        AVG(LOSS_BY_KO_TKO) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND 1 PRECEDING
        ) AS LOSS_RATE_BY_KO_TKO, 
        AVG(LOSS_BY_SUBMISSION) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND 1 PRECEDING
        ) AS LOSS_RATE_BY_SUBMISSION, 
        AVG(LOSS_BY_DECISION) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND 1 PRECEDING
        ) AS LOSS_RATE_BY_DECISION, 
        TOTAL_FIGHTS, 
        SUM(TOTAL_TIME_SECONDS) OVER (
          PARTITION BY UFCSTATS_FIGHTER_ID 
          ORDER BY 
            FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
            AND 1 PRECEDING
        ) AS TIME_SECONDS_FOUGHT 
      FROM 
        stacked_sherdog_raw
    ) 
  WHERE 
    EVENT_ID IN (
      SELECT 
        EVENT_ID 
      FROM 
        sherdog.SHERDOG_BOUTS
    )
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
ufcstats_sherdog_merged_temp AS (
  SELECT 
    t1.BOUT_ID, 
    t1.FIGHTER_ID, 
    t1.FIGHTER_BOUT_NUMBER, 
    t2.AGE_DAYS, 
    t2.AGE_DAYS_AVERAGE, 
    t2.DAYS_SINCE_DEBUT, 
    t2.DAYS_SINCE_LAST_FIGHT_AVERAGE, 
    t2.WINS, 
    t2.WIN_RATE, 
    t2.LOSS_RATE_BY_KO_TKO, 
    t2.LOSS_RATE_BY_SUBMISSION, 
    t2.LOSS_RATE_BY_DECISION, 
    t2.TOTAL_FIGHTS, 
    t2.TIME_SECONDS_FOUGHT 
  FROM 
    bout_num_by_fighter AS t1 
    INNER JOIN stacked_sherdog_features AS t2 ON t1.FIGHTER_ID = t2.UFCSTATS_FIGHTER_ID 
    AND t1.FIGHTER_BOUT_NUMBER = t2.FIGHTER_BOUT_NUMBER
) 
SELECT 
  t1.BOUT_ID, 
  t1.EVENT_ID, 
  t1.DATE, 
  t1.BOUT_ORDINAL, 
  t2.AGE_DAYS - t3.AGE_DAYS AS SHERDOG_AGE_DAYS_DIFF, 
  t2.AGE_DAYS_AVERAGE - t3.AGE_DAYS_AVERAGE AS SHERDOG_AGE_DAYS_AVERAGE_DIFF, 
  t2.DAYS_SINCE_DEBUT - t3.DAYS_SINCE_DEBUT AS SHERDOG_DAYS_SINCE_DEBUT_DIFF, 
  t2.DAYS_SINCE_LAST_FIGHT_AVERAGE - t3.DAYS_SINCE_LAST_FIGHT_AVERAGE AS SHERDOG_DAYS_SINCE_LAST_FIGHT_AVERAGE_DIFF, 
  t2.WINS - t3.WINS AS SHERDOG_WINS_DIFF, 
  t2.WIN_RATE - t3.WIN_RATE AS SHERDOG_WIN_RATE_DIFF, 
  t2.LOSS_RATE_BY_KO_TKO - t3.LOSS_RATE_BY_KO_TKO AS SHERDOG_LOSS_RATE_BY_KO_TKO_DIFF, 
  t2.LOSS_RATE_BY_SUBMISSION - t3.LOSS_RATE_BY_SUBMISSION AS SHERDOG_LOSS_RATE_BY_SUBMISSION_DIFF, 
  t2.LOSS_RATE_BY_DECISION - t3.LOSS_RATE_BY_DECISION AS SHERDOG_LOSS_RATE_BY_DECISION_DIFF, 
  t2.TOTAL_FIGHTS - t3.TOTAL_FIGHTS AS SHERDOG_TOTAL_FIGHTS_DIFF, 
  t2.TIME_SECONDS_FOUGHT - t3.TIME_SECONDS_FOUGHT AS SHERDOG_TIME_SECONDS_FOUGHT_DIFF, 
  CASE t1.RED_OUTCOME WHEN 'W' THEN 1 WHEN 'L' THEN 0 ELSE NULL END AS RED_WIN 
FROM 
  main.UFCSTATS_BOUTS_OVERALL AS t1 
  INNER JOIN ufcstats_sherdog_merged_temp AS t2 ON t1.BOUT_ID = t2.BOUT_ID 
  AND t1.RED_FIGHTER_ID = t2.FIGHTER_ID 
  INNER JOIN ufcstats_sherdog_merged_temp AS t3 ON t1.BOUT_ID = t3.BOUT_ID 
  AND t1.BLUE_FIGHTER_ID = t3.FIGHTER_ID 
WHERE 
  t2.FIGHTER_BOUT_NUMBER > 1 
  AND t3.FIGHTER_BOUT_NUMBER > 1 
  AND t1.DATE >= ? 
ORDER BY 
  t1.DATE, 
  t1.EVENT_ID, 
  t1.BOUT_ORDINAL;


-- Winning/losing streaks
WITH sherdog_streak_ids AS (
  SELECT 
    t2.UFCSTATS_FIGHTER_ID, 
    t1.EVENT_ID, 
    t1.FIGHTER_BOUT_ORDINAL, 
    t1.OUTCOME, 
    (
      (t1.FIGHTER_BOUT_ORDINAL + 1) - ROW_NUMBER() OVER (
        PARTITION BY t1.FIGHTER_ID, 
        t1.OUTCOME 
        ORDER BY 
          t1.FIGHTER_BOUT_ORDINAL
      )
    ) AS rn_diff 
  FROM 
    sherdog.SHERDOG_BOUT_HISTORY AS t1 
    INNER JOIN sherdog.SHERDOG_FIGHTER_LINKAGE AS t2 ON t1.FIGHTER_ID = t2.SHERDOG_FIGHTER_ID
), 
sherdog_streaks AS (
  SELECT 
    UFCSTATS_FIGHTER_ID, 
    EVENT_ID, 
    FIGHTER_BOUT_ORDINAL, 
    OUTCOME, 
    ROW_NUMBER() OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID, 
      rn_diff 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL
    ) AS streak 
  FROM 
    sherdog_streak_ids
), 
sherdog_win_lose_streaks AS (
  SELECT 
    UFCSTATS_FIGHTER_ID, 
    EVENT_ID, 
    FIGHTER_BOUT_ORDINAL, 
    OUTCOME, 
    CASE WHEN OUTCOME = 'W' THEN streak ELSE 0 END AS WINNING_STREAK_POST, 
    CASE WHEN OUTCOME = 'L' THEN streak ELSE 0 END AS LOSING_STREAK_POST 
  FROM 
    sherdog_streaks
), 
sherdog_streak_features AS (
  SELECT 
    UFCSTATS_FIGHTER_ID, 
    EVENT_ID, 
    FIGHTER_BOUT_ORDINAL, 
    LAG(WINNING_STREAK_POST, 1) OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL
    ) AS WINNING_STREAK, 
    LAG(LOSING_STREAK_POST, 1) OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL
    ) AS LOSING_STREAK, 
    AVG(WINNING_STREAK_POST) OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
        AND 1 PRECEDING
    ) AS WINNING_STREAK_AVERAGE, 
    AVG(LOSING_STREAK_POST) OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
        AND 1 PRECEDING
    ) AS LOSING_STREAK_AVERAGE, 
    MAX(WINNING_STREAK_POST) OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
        AND 1 PRECEDING
    ) AS WINNING_STREAK_MAX, 
    MAX(LOSING_STREAK_POST) OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL ROWS BETWEEN UNBOUNDED PRECEDING 
        AND 1 PRECEDING
    ) AS LOSING_STREAK_MAX 
  FROM 
    sherdog_win_lose_streaks
), 
sherdog_streak_features_filtered AS (
  SELECT 
    *, 
    ROW_NUMBER() OVER (
      PARTITION BY UFCSTATS_FIGHTER_ID 
      ORDER BY 
        FIGHTER_BOUT_ORDINAL
    ) AS FIGHTER_BOUT_NUMBER 
  FROM 
    sherdog_streak_features 
  WHERE 
    EVENT_ID IN (
      SELECT 
        EVENT_ID 
      FROM 
        sherdog.SHERDOG_BOUTS
    )
), 
sherdog_streak_features_filtered_col_hack AS (
  SELECT 
    *, 
    ROW_NUMBER() OVER(
      ORDER BY 
        UFCSTATS_FIGHTER_ID, 
        FIGHTER_BOUT_NUMBER
    ) AS col_hack 
  FROM 
    sherdoG_streak_features_filtered
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
bout_num_by_fighter_col_hack AS (
  SELECT 
    *, 
    ROW_NUMBER() OVER(
      ORDER BY 
        FIGHTER_ID, 
        FIGHTER_BOUT_NUMBER
    ) AS col_hack 
  FROM 
    bout_num_by_fighter
), 
ufcstats_streaks_merged_temp AS (
  SELECT 
    t1.BOUT_ID, 
    t1.FIGHTER_ID, 
    t1.FIGHTER_BOUT_NUMBER, 
    t2.WINNING_STREAK, 
    t2.LOSING_STREAK, 
    t2.WINNING_STREAK_AVERAGE, 
    t2.LOSING_STREAK_AVERAGE, 
    t2.WINNING_STREAK_MAX, 
    t2.LOSING_STREAK_MAX 
  FROM 
    bout_num_by_fighter_col_hack AS t1 
    LEFT JOIN sherdog_streak_features_filtered_col_hack AS t2 ON t1.col_hack = t2.col_hack
) 
SELECT 
  t1.BOUT_ID, 
  t1.EVENT_ID, 
  t1.DATE, 
  t1.BOUT_ORDINAL, 
  t2.WINNING_STREAK - t3.WINNING_STREAK AS SHERDOG_WINNING_STREAK_DIFF, 
  t2.LOSING_STREAK - t3.LOSING_STREAK AS SHERDOG_LOSING_STREAK_DIFF, 
  t2.WINNING_STREAK_AVERAGE - t3.WINNING_STREAK_AVERAGE AS SHERDOG_WINNING_STREAK_AVERAGE_DIFF, 
  t2.LOSING_STREAK_AVERAGE - t3.LOSING_STREAK_AVERAGE AS SHERDOG_LOSING_STREAK_AVERAGE_DIFF, 
  t2.WINNING_STREAK_MAX - t3.WINNING_STREAK_MAX AS SHERDOG_WINNING_STREAK_MAX_DIFF, 
  t2.LOSING_STREAK_MAX - t3.LOSING_STREAK_MAX AS SHERDOG_LOSING_STREAK_MAX_DIFF, 
  CASE t1.RED_OUTCOME WHEN 'W' THEN 1 WHEN 'L' THEN 0 ELSE NULL END AS RED_WIN 
FROM 
  main.UFCSTATS_BOUTS_OVERALL AS t1 
  INNER JOIN ufcstats_streaks_merged_temp AS t2 ON t1.BOUT_ID = t2.BOUT_ID 
  AND t1.RED_FIGHTER_ID = t2.FIGHTER_ID 
  INNER JOIN ufcstats_streaks_merged_temp AS t3 ON t1.BOUT_ID = t3.BOUT_ID 
  AND t1.BLUE_FIGHTER_ID = t3.FIGHTER_ID 
WHERE 
  t2.FIGHTER_BOUT_NUMBER > 1 
  AND t3.FIGHTER_BOUT_NUMBER > 1 
  AND t1.DATE >= ? 
ORDER BY 
  t1.DATE, 
  t1.EVENT_ID, 
  t1.BOUT_ORDINAL;