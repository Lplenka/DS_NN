#https://leetcode.com/articles/department-top-three-salaries/
use amazon;
SET SQL_SAFE_UPDATES = 0;
create table if not exists Employees
(
Id int,
Name_ varchar(225),
Salary int,
DepartmentId int
);
delete from Employees;
insert into Employees(Id, Name_, Salary, DepartmentId)
values
(1, "Joe", 85000, 1),
(2, "Henry", 80000, 2),
(3, "Sam", 60000, 2),
(4, "Max", 90000, 1),
(5, "Janet", 69000, 1),
(6, "Randy", 85000, 1),
(7, "Will", 70000, 1);


create table if not exists Department 
(
Id int,
Name varchar(225)
);

delete from Department;
insert into Department (Id, Name)
values
(1, "IT"),
(2, "Sales");

select * from Employees;

select * from Department;


#select top three salaries from one department
select e1.Salary from Employees e1
where 3 >

(
select count( distinct e2.salary) from Employees e2
where 
e2.salary > e1.salary
);

-- select distinct e1.Salary from Employees e1
-- where 3 >
-- 
-- (
-- select count( distinct e2.salary) from Employees e2
-- where 
-- e2.salary > e1.salary
-- );

#department-wise-top-three-salaries/
# Solution I
select d.Name, e1.Name_, e1.Salary
from Employees e1
join Department d
on e1.DepartmentId = d.Id
where 3 >
(
select count( distinct e2.salary) from Employees e2
where 
e2.salary > e1.salary
AND e1.DepartmentId = e2.DepartmentId
);
# Solution II
select d.Name as Department, a. Name_ as Employees, a. Salary 
from 
(
select e.*, DENSE_RANK() OVER (partition by DepartmentId order by e.Salary desc) as DeptPayRank 
from Employees e
) a 
join Department d
on a.DepartmentId = d. Id 
where DeptPayRank <=3; 

