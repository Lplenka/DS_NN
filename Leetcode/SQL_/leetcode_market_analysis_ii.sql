#https://leetcode.jp/problemdetail.php?id=1159
#https://github.com/openset/leetcode/blob/b89c05d24bf9bfdff7c87f11d8d165e2221de59f/problems/market-analysis-ii/

use amazon;
SET SQL_SAFE_UPDATES = 0;

 
Create table If Not Exists Users1 (user_id int, join_date date, favorite_brand varchar(10));
create table if not exists Orders1 (order_id int, order_date date, item_id int, buyer_id int, seller_id int);
create table if not exists Items1 (item_id int, item_brand varchar(10));
Truncate table Users1;
insert into Users1 (user_id, join_date, favorite_brand) values ('1', '2019-01-01', 'Lenovo');
insert into Users1 (user_id, join_date, favorite_brand) values ('2', '2019-02-09', 'Samsung');
insert into Users1 (user_id, join_date, favorite_brand) values ('3', '2019-01-19', 'LG');
insert into Users1 (user_id, join_date, favorite_brand) values ('4', '2019-05-21', 'HP');
Truncate table Orders1;
insert into Orders1 (order_id, order_date, item_id, buyer_id, seller_id) values ('1', '2019-08-01', '4', '1', '2');
insert into Orders1 (order_id, order_date, item_id, buyer_id, seller_id) values ('2', '2019-08-02', '2', '1', '3');
insert into Orders1 (order_id, order_date, item_id, buyer_id, seller_id) values ('3', '2019-08-03', '3', '2', '3');
insert into Orders1 (order_id, order_date, item_id, buyer_id, seller_id) values ('4', '2019-08-04', '1', '4', '2');
insert into Orders1 (order_id, order_date, item_id, buyer_id, seller_id) values ('5', '2019-08-04', '1', '3', '4');
insert into Orders1 (order_id, order_date, item_id, buyer_id, seller_id) values ('6', '2019-08-05', '2', '2', '4');
Truncate table Items1;
insert into Items1 (item_id, item_brand) values ('1', 'Samsung');
insert into Items1 (item_id, item_brand) values ('2', 'Lenovo');
insert into Items1 (item_id, item_brand) values ('3', 'LG');
insert into Items1 (item_id, item_brand) values ('4', 'HP');


 
#SOLUTION

select distinct a.user_id , ifnull(b.s_f, "no") as "2nd _item_fav_brand"

from Users1 a

left join

(
 
	select user_id, s_f 
	from
	( select tu.order_id, tu.order_date,  tu.item_id, tu.buyer_id, tu.seller_id,
	tu.rank_, tu.user_id, tu.favorite_brand, i.item_brand, 
		case 
		when i.item_brand = tu.favorite_brand and tu.rank_=2 
		then "yes" 
		else "no" 
		end as s_f from 
		(
		select * from 
			(select *, dense_rank() over w as rank_
			 from Orders1
			 window w as (partition by seller_id order by order_date)) t
		right join 
		Users1 u
		on u.user_id = t.seller_id
		) tu
		left join 
		Items1 i
		on i.item_id = tu.item_id
		order by order_date
	) x
	where x.rank_ = 2
	order by user_id
    
) as b

on a.user_id=b.user_id;

 
 
 #Solution II
 select user_id as seller_id, 
       if(isnull(item_brand), "no", "yes") as 2nd_item_fav_brand
from Users left join
(select seller_id, item_brand
from Orders inner join Items
on Orders.item_id = Items.item_id
where (seller_id, order_date) in
(select seller_id, min(order_date) as order_date
 from Orders
 where (seller_id, order_date) not in
 (select seller_id, min(order_date) from Orders group by seller_id)
group by seller_id)
 ) as t
on Users.user_id = t.seller_id and favorite_brand = item_brand