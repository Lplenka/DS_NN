use amazon;

create table IF NOT EXISTS Employee 
(
id int NOT NULL AUTO_INCREMENT,
salary int,
PRIMARY KEY (id)
);

insert into Employee (salary)
values 
(5000),(4000),(2000),(2345),(5677),(2210),
(7890),(2345),(10000),(4654),(9992),(2324);


select * from Employee;

# Median Salary
select AVG(e.salary)
from 
(
  select @rowindex := @rowindex+1 as rank,  salary  
  from Employee, (select @rowindex := -1) r
  order by Employee.salary


) as e
where 
e.rank IN (FLOOR(@rowindex/ 2) , CEIL(@rowindex  / 2));

# Median Salary II



#Rank the distinct salaries
select @rowindex := @rowindex+1 as rank, salary 
from ( select distinct salary as salary from Employee) f,
 (select @rowindex := 0) r
order by f.salary;

#check version
SHOW VARIABLES LIKE '%version%';

#Nth Highest salary
select t.salary from
(
select @rowindex := @rowindex+1 as rank, Employee.salary as salary  
from Employee, (select @rowindex := 0) r
order by Employee.salary desc
) t

where t.rank = 3;

#https://leetcode.com/problems/nth-highest-salary/

-- CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
--  BEGIN
-- set @rowindex := 0;
-- RETURN (
--  
-- 
--     
-- select t.Salary as salary from
-- (
--     select @rowindex := @rowindex+1 as rank_, 
--     Salary from
--     (select distinct Salary from Employee) as p
--     order by Salary desc
-- ) t
-- 
-- where t.rank_ = N
--       
-- 
--       
--       
-- );
-- 
-- END

-- select distinct salary
-- from
-- (
-- select salary, dense_rank() over (order by Salary desc) as Rank
-- from Employee
-- ) p
-- where p.Rank = @N
-- );