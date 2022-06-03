CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      # Write your MySQL query statement below.
      
      select distinct(T1.salary) from
      
      (select salary, dense_rank() over w as 'dense_rank'
      from Employee 
      window w as (order by salary desc)) as T1
      where T1.dense_rank = N
  );
END