-- user continuous showup date and days
-- 就是首先根据登录日期排序，这样我们获得的排序序号就是连续的，然后再统计每个登录日期和一个初始日期的间隔天数，如果登录连续的话，2个值相减之后也可以用来判断是否连续。
	(SELECT [user_id]
		  ,[login_date]
		  , RANK() over (partition by user_id order by user_id, login_date) as rnk
		  , lag(login_date) over (partition by user_id order by login_date) as dlag
		  , -DATEDIFF(day, login_date, lag(login_date) over (partition by user_id order by login_date)) as dd
		  , -datediff(day, login_date, min(login_date) over (partition by user_id)) as dd0
		  , RANK() over (partition by user_id order by user_id, login_date) 
		   + datediff(day, login_date, min(login_date) over (partition by user_id)) as ddrnk
		--, DATEDIFF('2020-10-10' , '2020-11-11')
	  FROM [test].[dbo].[tm_login_log])
;

SELECT user_id, 
	min(login_date) as start_, 
	max(login_date) as end_, 
	count(ddrnk) as days
FROM
	(SELECT [user_id]
		  ,[login_date]
		  , RANK() over (partition by user_id order by user_id, login_date) as rnk
		  , lag(login_date) over (partition by user_id order by login_date) as dlag
		  , -DATEDIFF(day, login_date, lag(login_date) over (partition by user_id order by login_date)) as dd
		  , -datediff(day, login_date, min(login_date) over (partition by user_id)) as dd0
		  , RANK() over (partition by user_id order by user_id, login_date) 
		   + datediff(day, login_date, min(login_date) over (partition by user_id)) as ddrnk
		--, DATEDIFF('2020-10-10' , '2020-11-11')
	  FROM [test].[dbo].[tm_login_log]) t
GROUP by user_id, ddrnk
HAVING count(ddrnk) > 1
ORDER by user_id, start_;

