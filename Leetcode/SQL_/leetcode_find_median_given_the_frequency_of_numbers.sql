#https://leetcode.jp/problemdetail.php?id=571

use amazon;
SET SQL_SAFE_UPDATES = 0;

Create table If Not Exists Numbers (Number int, Frequency int);
Truncate table Numbers;
insert into Numbers (Number, Frequency) values ('0', '7');
insert into Numbers (Number, Frequency) values ('1', '1');
insert into Numbers (Number, Frequency) values ('2', '3');
insert into Numbers (Number, Frequency) values ('3', '1');

select * from Numbers;

select t1.Number, 
        sum(case when t1.Number>t2.Number then t2.Frequency else 0 end),
            sum(case when t1.Number<t2.Number then t2.Frequency else 0 end)
    from Numbers as t1, Numbers as t2
    group by t1.Number;

select t1.Number, 
        abs(sum(case when t1.Number>t2.Number then t2.Frequency else 0 end) -
            sum(case when t1.Number<t2.Number then t2.Frequency else 0 end)) as count_diff
    from Numbers as t1, Numbers as t2
    group by t1.Number;
    
#final solution
#basic idea is if your count is more than or equal to 
#those numbers which are less or more than you, then sayad tum median ho.

/*So in general, the median's frequency should be equal or greater than the absolute
 difference of its bigger elements and small ones in an array no matter whether 
 it has odd or even amount of numbers and whether they are distinct. 
This rule is the key, and it is represented as the following code. */

select avg(t3.Number) as median 
from Numbers as t3 
inner join 
    (select t1.Number, 
        abs(sum(case when t1.Number>t2.Number then t2.Frequency else 0 end) -
            sum(case when t1.Number<t2.Number then t2.Frequency else 0 end)) as count_diff
    from numbers as t1, numbers as t2
    group by t1.Number) as t4
on t3.Number = t4.Number
where t3.Frequency>=t4.count_diff