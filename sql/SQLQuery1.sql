use [real_estate];

select top 10 * from Restaurant_customer, Restaurant_feature, Restaurant_rating;
select top 10 * FROM Restaurant_customer;
select top 10 * FROM Restaurant_feature;

select distinct birth_year, budget, count(userid) over (partition by budget, birth_year) uid FROM Restaurant_customer order by birth_year desc;
select birth_year, budget, count(userid) uid FROM Restaurant_customer group by birth_year, budget order by birth_year desc;

select budget, count(userid) * 1.0 / (select count(userid) from Restaurant_customer) AS uid from Restaurant_customer group by budget order by COUNT(userid);

select top 1 * from test;

select drink_level, AVG(weight / height) as rho from Restaurant_customer group by drink_level ORDER BY rho DESC;
select birth_year, drink_level, count(drink_level)  over (partition by birth_year) from Restaurant_customer --;order by count(drink_level);

select d1, d2, day(d2 - d1) as d12, DATEDIFF(DAY, d2, d1) AS d3, 
DATEDIFF(DAY, '2017/08/20 07:00', '2017/08/25 12:45') AS dt4  
from (select GETDATE() as d1, GETDATE() - 30 as d2) t;

SELECT DATEPART(year, '2017/08/25') AS DatePartInt;
  SELECT DATEPART(yy, '12/1/1993') AS DatePartInt;
  SELECT DATEPART(yy, ' 08/25/2010 ') AS DatePartInt;

  select dealid, lag(dealid, 1) over (order by dealid) as offset from deals order by DealID;

-- exchange neigbour rows
SELECT T2.placeID, t2.name, t2.row_num 
FROM 
(SELECT t1.placeid, t1.name, ROW_NUMBER() over (order by t1.placeid) as row_num 
from  Restaurant_feature t1) as t2
ORDER BY --id_new; --T2.row_num;
    CASE WHEN (t2.row_num % 2) = 0 THEN t2.row_num-1 
    ELSE t2.row_num + 1 END --AS id_new  
;

select * from 
(SELECT row_number() over (order by t1.placeID) as row_num, t1.placeid, t1.name
from Restaurant_feature t1 ) t2
-- where t2.row_num % 2 = 0
order by t2.placeID;

-- median by group
select * 
FROM 
(select transport, height, rank() over (partition by transport order by height) as row_num, count(height) over (partition by transport) cnt
FROM Restaurant_customer ) t1
WHERE row_num = (cnt / 2) ;

/****** Script for SelectTopNRows command from SSMS  ******/

select top 20 * from Restaurant_customer r
where r.[userID] not in (
	SELECT TOP (10) [userID]
  FROM [real_estate].[dbo].[Restaurant_customer]);

--  select top 20 * from Restaurant_customer r
--	where not exists (
--	SELECT TOP (10) [userID]
--  FROM [real_estate].[dbo].[Restaurant_customer] b
--  where r.userid=b.userID);

  select top 20 * from Restaurant_customer r
   except 
	(SELECT TOP (10) *
  FROM [real_estate].[dbo].[Restaurant_customer] b) 


select top 20 *, rank() over (partition by r.drink_level order by r.userid) rnk from Restaurant_customer r
left join 
	(SELECT TOP (10) b.[userID]
  FROM [real_estate].[dbo].[Restaurant_customer] b) as c
  on r.userID=c.userid 
  where c.userID is null;

select distinct r.drink_level, sum(height) over (partition by r.drink_level) as sum_drink from Restaurant_customer r;

  with s(a,b, c) as (select rand(11), isnull(null, 0), 'a, b')
  select * from s;

  select 10 / (select 1);
 