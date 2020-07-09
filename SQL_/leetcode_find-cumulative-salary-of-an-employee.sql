#https://leetcode.com/articles/find-cumulative-salary-of-an-employee/
use amazon;
SET SQL_SAFE_UPDATES = 0;
create table if not exists Employee2
(
Id int,
Month varchar(225),
Salary int
);

delete from Employee2;

insert into Employee2 ( Id, Month, Salary)
values
	(1, 1, 20),
	(2, 1, 20),
	(1, 2, 30),
	(2, 2, 30),
	(3, 2, 40),
	(1, 3, 40),
	(3, 3, 60),
	(1, 4, 60),
	(3, 4, 70);
    
select * from Employee2;

#Sub Solution    
select e1.Id, e1.Month, e1.Salary, Salary
from 
(select Id, Month, Salary from Employee2) e1
 inner join
(select Id, max(Month) as m from Employee2 group by Id) e2
on e1.Id = e2.Id
where e1.Month < e2.m
order by e1.Id, e1.Month desc;


# Solution
select t1.Id, t1.Month, SUM(t2.Salary) as Sal
from

(

    select e1.Id, e1.Month, e1.Salary
    from 
    (select Id, Month, Salary from Employee2) e1
     inner join
    (select Id, max(Month) as m from Employee2 group by Id) e2
    on e1.Id = e2.Id
    where e1.Month < e2.m
    order by e1.Id, e1.Month desc


) t1


inner join 

(
    select e1.Id, e1.Month, e1.Salary
    from 
    (select Id, Month, Salary from Employee2) e1
     inner join
    (select Id, max(Month) as m from Employee2 group by Id) e2
    on e1.Id = e2.Id
    where e1.Month < e2.m
    order by e1.Id, e1.Month desc

) t2 

on t1.Id = t2.Id where t1.Month >= t2.Month

group by t1.Id, t1.Month
order by t1.Id , t1.Month desc;
