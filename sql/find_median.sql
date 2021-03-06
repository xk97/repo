/****** Script for SelectTopNRows command from SSMS  ******/
SELECT TOP (1000) [ClientID]
      ,[Client_City]
      ,[Client_State]
      ,[Client_Zip]
      ,[Client_CountryID]
      ,[Client_NationalAccountName]
      ,[Client_NATypeID]
      ,[Client_IsForeign]
      ,[Client_ThirdPartyCategoryID]
      ,[Client_InvestorType]
      ,[Client_isUmbrella]
      ,[Client_isDeleted]
      ,[Client_SalesForceID]
      ,[Client_LenderBook]
      ,[Client_BuyerBook]
      ,[Client_SFBook]
      ,[Client_NSBook]
      ,[Client_HFFSBook]
  FROM [real_estate].[dbo].[clients];

  -- find median of column clientid
 with tc as (
	select 
	clientid, client_state,
	ROW_NUMBER() over (order by clientid) as r
	,(select count(*) from clients) as n
	from clients
)
select clientid, client_state, tc.n, tc.r
  from tc
  where tc.r in ((tc.n + 1) / 2, (tc.n + 2) / 2)
  ;

  -- median clientid by state
  with tc as (
	select 
	clientid, client_state,
	ROW_NUMBER() over (partition by client_state order by clientid) as r
	, count(*) over (partition by client_state) as n
	from clients
	where Client_State is not null
)
select clientid, client_state, tc.n, tc.r
  from tc
  where tc.r in ((tc.n + 1) / 2, (tc.n + 2) / 2)
  ;

  select distinct Client_State, client_zip
  from clients
  where client_state is not null
	and client_zip is not null
	and client_zip != 'NULL';

  select distinct Client_State
  , max(client_zip) over (partition by client_state) as mx
  , min(client_zip) over (partition by client_state) as mn
  from clients
  where client_state is not null
	and client_zip != 'NULL'
	and client_zip is not null;