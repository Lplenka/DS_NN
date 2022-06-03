#https://leetcode.jp/problemdetail.php?id=618
/*
https://github.com/openset/leetcode/blob/08c1f39a539ebdc003f16269cfca5868b36ae8af/
problems/students-report-by-geography/students_report_by_geography.sql
*/
use amazon;
SET SQL_SAFE_UPDATES = 0;

Create table If Not Exists student (name varchar(50), continent varchar(7));
Truncate table student;
insert into student (name, continent) values ('Jane', 'America');
insert into student (name, continent) values ('Pascal', 'Europe');
insert into student (name, continent) values ('Xi', 'Asia');
insert into student (name, continent) values ('Jack', 'America');