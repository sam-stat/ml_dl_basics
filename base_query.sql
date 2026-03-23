create or replace table datascience.temp.sriram_share_data_consistent_base as

  with loc_monthly_snapshot as (                                                                                                                                                                
  select snap.uuid as user_id,
  date_format(snap.snapshot_date, 'yyyyMM') as statement_month,                                                                                                                                 
  snap.approved_limit,                                                                                                                                                                          
  round(snap.interest_rate, 2) as roi,
  snap.banking_vendor,                                                                                                                                                                          
  count(distinct snap.snapshot_date) as active_days                                                                                                                                           
  from analytics.cash.cash_loc_snapshot_v2 snap
  where snap.status in ('CREATED', 'APPROVED', 'OPEN') and
        snap.snapshot_type = 'DAY' and
        snap.snapshot_date >= '2023-04-01' and
        snap.snapshot_date <= '2025-09-30'
  group by 1, 2, 3, 4, 5
  having count(case when snap.status in ('CREATED', 'APPROVED') then 1 end) > 0),

  /*
  Pick the offer with the most active days per user-month
  Filter to whitelisted users with consistent offer > 20 days
  */
  loc_flags as (
  select user_id,
  statement_month,                                                                                                                                                                              
  active_days,
  approved_limit as max_approved_limit,                                                                                                                                                         
  roi                                                                                                                                                                                         
  from loc_monthly_snapshot
  where active_days > 20
  qualify row_number() over (partition by user_id, statement_month order by active_days desc) = 1),

  /*
  Pick 100K users per month independently from all eligible users in that month.
  Each month is sampled separately — no cross-month user exclusion.
  Deterministic random ordering via md5(concat(user_id, statement_month))
  includes statement_month so the same user gets a different rank each month.
  */                                                                                                                                                                                            
  unique_month_users as (
  select user_id,
  statement_month,
  active_days,
  max_approved_limit,
  roi,
  row_number() over (partition by statement_month order by md5(concat(user_id, statement_month))) as month_rank
  from loc_flags),

  /*
  Flatten loan history to user + disbursal_month (yyyyMM) for easy joining.
  */
  loans_monthly as (
  select user_id,
  date_format(disbursal_date, 'yyyyMM') as disbursal_month
  from lending.default.sh_all_loans_2),

  /*
  Flag 1 (is_repeat_borrower): 1 if user had any loan before this statement_month — open-state user.
  Flag 2 (has_loan_in_month): 1 if user took a loan in this exact statement_month.
  */
  user_loan_flags as (
  select users.user_id,
  users.statement_month,
  users.active_days,
  users.max_approved_limit,
  users.roi,
  max(case when loan.disbursal_month < users.statement_month then 1 else 0 end) as is_repeat_borrower,
  max(case when loan.disbursal_month = users.statement_month then 1 else 0 end) as has_loan_in_month
  from unique_month_users users
  left join loans_monthly loan on (loan.user_id = users.user_id)
  where users.month_rank <= 100000
  group by 1, 2, 3, 4, 5)

  select * from user_loan_flags