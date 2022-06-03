# Write your ML uery statement below


select COALESCE(   
(select distinct(T1.salary)  from
(select salary, dense_rank() over w as 'dense_rank' from Employee
window w as (order by salary desc)) T1
where T1.dense_rank = 2), NULL) SecondHighestSalary;


# SELECT Salary FROM Employee e 
# WHERE 2=(SELECT COUNT(DISTINCT Salary) 
# FROM Employee p 
# WHERE e.Salary<=p.Salary);