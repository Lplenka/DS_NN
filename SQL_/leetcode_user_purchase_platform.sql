#https://leetcode.jp/problemdetail.php?id=1127

use amazon;
SET SQL_SAFE_UPDATES = 0;

Create table If Not Exists Spending (user_id int, spend_date date, platform ENUM('desktop', 'mobile'), amount int);
Truncate table Spending;
insert into Spending (user_id, spend_date, platform, amount) values ('1', '2019-07-01', 'mobile', '100');
insert into Spending (user_id, spend_date, platform, amount) values ('1', '2019-07-01', 'desktop', '100');
insert into Spending (user_id, spend_date, platform, amount) values ('2', '2019-07-01', 'mobile', '100');
insert into Spending (user_id, spend_date, platform, amount) values ('2', '2019-07-02', 'mobile', '100');
insert into Spending (user_id, spend_date, platform, amount) values ('3', '2019-07-01', 'desktop', '100');
insert into Spending (user_id, spend_date, platform, amount) values ('3', '2019-07-02', 'desktop', '100');




select spend_date, platform, case when sum(amount)>0 then sum(amount) else 0 end  as total_amount , count(distinct user_id) total_users 
from

(
select a.spend_date, b.platform from Spending a, ( select platform from Spending union select "both" as platform)b
group by a.spend_date, b.platform
) 
y 
left join
(
select t1.spend_date, t1.user_id, 
case 
when cnt > 1 then "both" 
else t2.platform
end as platform,
t2.amount
from 
(
select spend_date, user_id, count(platform) as cnt
from Spending
group by spend_date, user_id
) t1
inner join 
(
select user_id,spend_date, platform, amount from Spending
) t2
using (user_id, spend_date) 

)x
using(spend_date, platform)
group by  spend_date, platform
order by spend_date;




