#https://leetcode.jp/problemdetail.php?id=1225
#https://leetcode.com/problems/report-contiguous-dates
use amazon;
SET SQL_SAFE_UPDATES = 0;
create table if not exists Failed
(
fail_date date

);

delete from Failed;
insert into Failed(fail_date)
values
("2018-12-28"),
("2018-12-29"),
("2019-01-04"),
("2019-01-05");

create table if not exists Succeeded	
(
success_date date

);

delete from Succeeded;
insert into Succeeded(success_date)
values
("2018-12-30"),
("2018-12-31"),
("2019-01-01"),
("2019-01-02"),
("2019-01-03"),
("2019-01-06");

-- SET GLOBAL sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));
-- 
-- SELECT @@sql_mode;
--set session sql_mode='STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION';

select  period_state, min(date) as start_date, max(date) as end_date
from (
		select period_state, date,
         @rank_ := case when @prev = period_state then @rank_ else @rank_+1 end as rank_,
         @prev := period_state as prev
    from (
        select 'failed' as period_state, fail_date as date
        from Failed
        where fail_date between '2019-01-01' and '2019-12-31'
        union
        select 'succeeded' as period_state, success_date as date
        from Succeeded
        where success_date between '2019-01-01' and '2019-12-31') t, 
        (select @rank_:=0, @prev:='') r
    order by date asc
    
) as tt
group by rank_, period_state
order by rank_;
