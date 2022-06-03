#https://leetcode.com/articles/average-salary-departments-vs-company/
use amazon;
SET SQL_SAFE_UPDATES = 0;

create table salary
(
id int not null auto_increment,
employee_id int,
amount int,
pay_date DATE,

PRIMARY KEY (id)	

);
delete from salary;

insert into salary (employee_id, amount, pay_date)
values
(1, 9000, "2017-03-31"),
(2, 6000, "2017-03-31"),
(3, 10000, "2017-03-31"),
(1, 7000, "2017-02-28"),
(2, 6000, "2017-02-28"),
(3, 8000, "2017-02-28");

create table employee1
(
employee_id int not null auto_increment,
department_id int,
primary key (employee_id)
);


delete from employee1;
insert into employee1 (employee_id, department_id)
values
(1, 1),
(2,2),
(3,2);

select * from salary;
select * from employee1;


--  select month(pay_date), department_id, 

--  	
--  
-- select avg(amount)  from salary group by month(pay_date);
-- select month(t.pay_date), t.department_id,
-- case comparison 
-- when t.amount > t.avg_dp_salary then "higher"
-- when t.amount < t.avg_dp_salary then "lower"
-- else "same"
-- end 
-- from 
-- (select s.employee_id, s.amount, s.pay_date, d.department_id, avg(s.amount) as avg_dp_salary 
-- from salary s
-- join employee1 d
-- on s.employee_id = d.employee_id
-- group by month(s.pay_date), d.department_id
-- )t;

-- select month(t.pay_date), t.department_id,
-- case comparison 
-- when t.amount > t.avg_dp_salary then "higher"
-- when t.amount < t.avg_dp_salary then "lower"
-- else "same"
-- end 
-- from 

select b.mb, b.department_id,
case  
when b.dep_avg > a.avg_comp_salary then "higher"
when b.dep_avg < a.avg_comp_salary then "lower"
else "same"
end as comparison

 from 
(   select date_format(s.pay_date,"%x-%m") as mb, d.department_id, 
    avg(s.amount) as dep_avg
    from salary s
    join employee1 d
    on s.employee_id = d.employee_id
    group by d.department_id, date_format(pay_date,"%x-%m")
) b 
join 
(   select avg(amount) as avg_comp_salary , date_format(pay_date,"%x-%m") as ma from salary group by date_format(pay_date,"%x-%m") 
)  a

on b.mb = a.ma;

select date_format(pay_date,"%x-%m") from salary;
