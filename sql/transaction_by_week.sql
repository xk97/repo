CREATE TABLE USER_
(
    USERID nCHAR(20),
    CITY nCHAR(30)
);
CREATE TABLE CITY
(
    USERID nCHAR(20),
    TRANSACTION_ nchar(20),
    amount real,
    date_ Datetime
);

WITH CITY_TRANSACTION (cityid, amount, year_, week_) AS (
    SELECT cityid, amount,
        DATEPART(year, date_) as year_,
        DATEPART(week, date_) as week_
    FROM USER_
        JOIN CITY
        ON USER_.CITYID = CITY.cityid
)
    SELECT ct1.cityid,
        100.0 * ct1.amount / ct2.amount as PCT
    FROM (
    SELECT cityid, sum(amount)
        FROM CITY_TRANSACTION
    GROUP BY cityid, year_, week_
    ) ct1
JOIN
    (
    SELECT cityid, sum(amount)
    FROM CITY_TRANSACTION
    GROUP BY year_, week_
) ct2
ON ct1.cityid = ct2.cityid
AND ct1.year_ = ct2.year_
AND ct1.week_ = ct2.week_
;

SELECT CITYID, 
100.0 * AMOUNT / SUM(AMOUNT) OVER (PARTItION BY year_, week_) PCT
FROM CITY_TRANSACTION;


SELECT cityid, 100.0 * city_week_one / city_week_total as PCT
FROM (
    SELECT 
    cityid,
    SUM(amount) OVER ( PARTITION BY year_, week_) as city_week_total,
    SUM(amount) OVER ( PARTITION BY cityid, year_, week_ ) as city_week_one
    FROM city_transaction
)
;