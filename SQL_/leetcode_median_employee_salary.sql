#https://code.dennyzhang.com/median-employee-salary
#https://leetcode.com/articles/median-employee-salary/

use amazon;
SET SQL_SAFE_UPDATES = 0;
Create table If Not Exists Employee3 (Id int, Company varchar(255), Salary int);
Truncate table Employee;
insert into Employee3 (Id, Company, Salary) values ('1', 'A', '2341');
insert into Employee3 (Id, Company, Salary) values ('2', 'A', '341');
insert into Employee3 (Id, Company, Salary) values ('3', 'A', '15');
insert into Employee3 (Id, Company, Salary) values ('4', 'A', '15314');
insert into Employee3 (Id, Company, Salary) values ('5', 'A', '451');
insert into Employee3 (Id, Company, Salary) values ('6', 'A', '513');
insert into Employee3 (Id, Company, Salary) values ('7', 'B', '15');
insert into Employee3 (Id, Company, Salary) values ('8', 'B', '13');
insert into Employee3 (Id, Company, Salary) values ('9', 'B', '1154');
insert into Employee3 (Id, Company, Salary) values ('10', 'B', '1345');
insert into Employee3 (Id, Company, Salary) values ('11', 'B', '1221');
insert into Employee3 (Id, Company, Salary) values ('12', 'B', '234');
insert into Employee3 (Id, Company, Salary) values ('13', 'C', '2345');
insert into Employee3 (Id, Company, Salary) values ('14', 'C', '2645');
insert into Employee3 (Id, Company, Salary) values ('15', 'C', '2645');
insert into Employee3 (Id, Company, Salary) values ('16', 'C', '2652');
insert into Employee3 (Id, Company, Salary) values ('17', 'C', '65');



#SOLUTION


SELECT
    ANY_VALUE(Employee.Id), Employee.Company, Employee.Salary
FROM
    Employee3 as Employee,
    Employee3 alias
WHERE
    Employee.Company = alias.Company
GROUP BY Employee.Company , Employee.Salary
HAVING SUM(CASE
    WHEN Employee.Salary = alias.Salary THEN 1
    ELSE 0
END) >= ABS(SUM(SIGN(Employee.Salary - alias.Salary)))
ORDER BY ANY_VALUE(Employee.Id);

#SOLUTION II after sorting
SELECT 
    Id, Company, Salary
FROM
    (
    SELECT 
        e.Id,
            e.Salary,
            e.Company,
            IF(@prev = e.Company, @Rank_:=@Rank_ + 1, @Rank_:=1) AS rank_,
            @prev:=e.Company
    FROM
        Employee3 e, (SELECT @Rank_:=0, @prev:=0) AS temp
    ORDER BY e.Company , e.Salary , e.Id) Ranking
        INNER JOIN
    (SELECT 
        COUNT(*) AS totalcount, Company AS name
    FROM
        Employee3 e2
    GROUP BY e2.Company) companycount ON companycount.name = Ranking.Company
WHERE
    Rank_ = FLOOR((totalcount + 1) / 2)
        OR Rank_ = FLOOR((totalcount + 2) / 2);

