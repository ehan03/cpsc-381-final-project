WITH bout_num_by_fighter AS (
  SELECT 
    *, 
    ROW_NUMBER() OVER(
      PARTITION BY FIGHTER_ID 
      ORDER BY 
        DATE, 
        EVENT_ID, 
        BOUT_ORDINAL
    ) AS BOUT_NUMBER 
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
bouts_test AS (
  SELECT 
    t1.BOUT_ID, 
    t1.EVENT_ID, 
    t1.RED_FIGHTER_ID, 
    t1.BLUE_FIGHTER_ID, 
    t1.BOUT_ORDINAL, 
    t1.DATE, 
    CASE t1.RED_OUTCOME WHEN 'W' THEN 1 WHEN 'L' THEN 0 ELSE NULL END AS RED_WIN 
  FROM 
    main.UFCSTATS_BOUTS_OVERALL AS t1 
    LEFT JOIN bout_num_by_fighter AS t2 ON t1.RED_FIGHTER_ID = t2.FIGHTER_ID 
    AND t1.BOUT_ID = t2.BOUT_ID 
    LEFT JOIN bout_num_by_fighter AS t3 ON t1.BLUE_FIGHTER_ID = t3.FIGHTER_ID 
    AND t1.BOUT_ID = t3.BOUT_ID 
  WHERE 
    t2.BOUT_NUMBER > 1 
    AND t3.BOUT_NUMBER > 1 
    AND t1.DATE >= ?
  ORDER BY 
    t1.DATE, 
    t1.BOUT_ORDINAL
) 
SELECT 
  t1.BOUT_ID,  
  t1.EVENT_ID, 
  t1.DATE,
  t1.BOUT_ORDINAL, 
  t5.FIGHTER_1_ODDS AS RED_FIGHTER_ODDS, 
  t5.FIGHTER_2_ODDS AS BLUE_FIGHTER_ODDS, 
  t1.RED_WIN 
FROM 
  bouts_test AS t1 
  INNER JOIN fightoddsio.FIGHTODDSIO_EVENT_LINKAGE AS t2 ON t1.EVENT_ID = t2.UFCSTATS_EVENT_ID 
  INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t3 ON t1.RED_FIGHTER_ID = t3.UFCSTATS_FIGHTER_ID 
  INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t4 ON t1.BLUE_FIGHTER_ID = t4.UFCSTATS_FIGHTER_ID 
  INNER JOIN fightoddsio.FIGHTODDSIO_BOUTS AS t5 ON t2.FIGHTODDSIO_EVENT_SLUG = t5.EVENT_SLUG 
  AND t3.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_1_ID 
  AND t4.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_2_ID 
UNION ALL 
SELECT 
  t1.BOUT_ID, 
  t1.EVENT_ID, 
  t1.DATE,
  t1.BOUT_ORDINAL, 
  t5.FIGHTER_2_ODDS AS RED_FIGHTER_ODDS, 
  t5.FIGHTER_1_ODDS AS BLUE_FIGHTER_ODDS, 
  t1.RED_WIN 
FROM 
  bouts_test AS t1 
  INNER JOIN fightoddsio.FIGHTODDSIO_EVENT_LINKAGE AS t2 ON t1.EVENT_ID = t2.UFCSTATS_EVENT_ID 
  INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t3 ON t1.RED_FIGHTER_ID = t3.UFCSTATS_FIGHTER_ID 
  INNER JOIN fightoddsio.FIGHTODDSIO_FIGHTER_LINKAGE AS t4 ON t1.BLUE_FIGHTER_ID = t4.UFCSTATS_FIGHTER_ID 
  INNER JOIN fightoddsio.FIGHTODDSIO_BOUTS AS t5 ON t2.FIGHTODDSIO_EVENT_SLUG = t5.EVENT_SLUG 
  AND t3.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_2_ID 
  AND t4.FIGHTODDSIO_FIGHTER_ID = t5.FIGHTER_1_ID 
ORDER BY 
  t1.DATE, 
  t1.EVENT_ID, 
  t1.BOUT_ORDINAL;