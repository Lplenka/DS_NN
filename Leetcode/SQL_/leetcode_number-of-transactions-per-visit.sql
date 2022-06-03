#https://leetcode.com/problems/number-of-transactions-per-visit/
#https://code.dennyzhang.com/number-of-transactions-per-visit/
/*https://github.com/openset/leetcode/blob/b89c05d24bf9bfdff7c87f11d8d165e2221de59f/
 problems/number-of-transactions-per-visit/mysql_schemas.sql */
#https://leetcode.jp/problemdetail.php?id=1336

use amazon;
SET SQL_SAFE_UPDATES = 0;

Create table If Not Exists Visits (user_id int, visit_date date);
Create table If Not Exists Transactions (user_id int, transaction_date date, amount int);
Truncate table Visits;
insert into Visits (user_id, visit_date) values ('1', '2020-01-01');
insert into Visits (user_id, visit_date) values ('2', '2020-01-02');
insert into Visits (user_id, visit_date) values ('12', '2020-01-01');
insert into Visits (user_id, visit_date) values ('19', '2020-01-03');
insert into Visits (user_id, visit_date) values ('1', '2020-01-02');
insert into Visits (user_id, visit_date) values ('2', '2020-01-03');
insert into Visits (user_id, visit_date) values ('1', '2020-01-04');
insert into Visits (user_id, visit_date) values ('7', '2020-01-11');
insert into Visits (user_id, visit_date) values ('9', '2020-01-25');
insert into Visits (user_id, visit_date) values ('8', '2020-01-28');
Truncate table Transactions;
insert into Transactions (user_id, transaction_date, amount) values ('1', '2020-01-02', '120');
insert into Transactions (user_id, transaction_date, amount) values ('2', '2020-01-03', '22');
insert into Transactions (user_id, transaction_date, amount) values ('7', '2020-01-11', '232');
insert into Transactions (user_id, transaction_date, amount) values ('1', '2020-01-04', '7');
insert into Transactions (user_id, transaction_date, amount) values ('9', '2020-01-25', '33');
insert into Transactions (user_id, transaction_date, amount) values ('9', '2020-01-25', '66');
insert into Transactions (user_id, transaction_date, amount) values ('8', '2020-01-28', '1');
insert into Transactions (user_id, transaction_date, amount) values ('9', '2020-01-25', '99');



select z.cnt as transaction_counts, coalesce(y.visits_count,0)

from
(

select a.transaction_counts, count(a.transaction_counts ) as visits_count
from
(select t1.user_id, t1.visit_date, count(if(t2.transaction_date = t1.visit_date, 1, NULL)) as transaction_counts
from Visits t1
left join
Transactions t2
on t2.user_id = t1.user_id
group by t1.user_id, t1.visit_date) a

group by a.transaction_counts
order by a.transaction_counts) y

 right outer join 


(
select @num := @num + 1 as cnt
from 
(
	select max(transaction_counts) as max_trans from  
	(select t1.user_id, t1.visit_date, count(if(t2.transaction_date = t1.visit_date, 1, NULL)) as transaction_counts
	from Visits t1
	left join
	Transactions t2
	on t2.user_id = t1.user_id
	group by t1.user_id, t1.visit_date)p
 )q,Visits, (select @num := -1) r 
 
 where @num< q.max_trans ) z
 
 on y.transaction_counts = z.cnt;


#https://stackoverflow.com/questions/4340793/how-to-find-gaps-in-sequential-numbering-in-mysql
#https://www.xaprb.com/blog/2005/12/06/find-missing-numbers-in-a-sequence-with-sql/