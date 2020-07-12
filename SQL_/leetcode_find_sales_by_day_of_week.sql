#https://leetcode.jp/problemdetail.php?id=1479

use amazon;
SET SQL_SAFE_UPDATES = 0;

Create table If Not Exists Orders2 (order_id int, customer_id int, order_date date, item_id varchar(30), quantity int);
Create table If Not Exists Items2 (item_id varchar(30), item_name varchar(30), item_category varchar(30));
Truncate table Orders2;
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('1', '1', '2020-06-01', '1', '10');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('2', '1', '2020-06-08', '2', '10');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('3', '2', '2020-06-02', '1', '5');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('4', '3', '2020-06-03', '3', '5');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('5', '4', '2020-06-04', '4', '1');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('6', '4', '2020-06-05', '5', '5');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('7', '5', '2020-06-05', '1', '10');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('8', '5', '2020-06-14', '4', '5');
insert into Orders2 (order_id, customer_id, order_date, item_id, quantity) values ('9', '5', '2020-06-21', '3', '5');
Truncate table Items2;
insert into Items2 (item_id, item_name, item_category) values ('1', 'LC Alg. Book', 'Book');
insert into Items2 (item_id, item_name, item_category) values ('2', 'LC DB. Book', 'Book');
insert into Items2 (item_id, item_name, item_category) values ('3', 'LC SmarthPhone', 'Phone');
insert into Items2 (item_id, item_name, item_category) values ('4', 'LC Phone 2020', 'Phone');
insert into Items2 (item_id, item_name, item_category) values ('5', 'LC SmartGlass', 'Glasses');
insert into Items2 (item_id, item_name, item_category) values ('6', 'LC T-Shirt XL', 'T-shirt');



select x.item_category, sum(x.Monday),sum(x.Tuesday),sum(x.Wednesday),
sum(x.Thursday),sum(x.Friday),sum(x.Saturday),sum(x.Sunday)
from
(
	select t.item_category,
	case when day_name = "Monday" then sum(quan) else 0 end as Monday,
	case when day_name = "Tuesday" then sum(quan) else 0 end as Tuesday,
	case when day_name = "Wednesday" then sum(quan) else 0 end as Wednesday,
	case when day_name = "Thursday" then sum(quan) else 0 end as Thursday,
	case when day_name = "Friday" then sum(quan) else 0 end as Friday,
	case when day_name = "Saturday" then sum(quan) else 0 end as Saturday,
	case when day_name = "Sunday" then sum(quan) else 0 end as Sunday

	from 
	(
		select o.order_id, ifnull(dayname(o.order_date), 'all') as day_name, o.item_id, ifnull(o.quantity,0) as quan, i.item_category
		from Orders2 o
		right join 
		Items2 i
		on o.item_id = i.item_id
	)t

	group by t.item_category, day_name
)x
group by x.item_category;
