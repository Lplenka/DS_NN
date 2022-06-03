#https://leetcode.jp/problemdetail.php?id=1412

use amazon;
SET SQL_SAFE_UPDATES = 0;

 
Create table If Not Exists Student (student_id int, student_name varchar(30));
Create table If Not Exists Exam (exam_id int, student_id int, score int);
Truncate table Student;
insert into Student (student_id, student_name) values ('1', 'Daniel');
insert into Student (student_id, student_name) values ('2', 'Jade');
insert into Student (student_id, student_name) values ('3', 'Stella');
insert into Student (student_id, student_name) values ('4', 'Jonathan');
insert into Student (student_id, student_name) values ('5', 'Will');
Truncate table Exam;
insert into Exam (exam_id, student_id, score) values ('10', '1', '70');
insert into Exam (exam_id, student_id, score) values ('10', '2', '80');
insert into Exam (exam_id, student_id, score) values ('10', '3', '90');
insert into Exam (exam_id, student_id, score) values ('20', '1', '80');
insert into Exam (exam_id, student_id, score) values ('30', '1', '70');
insert into Exam (exam_id, student_id, score) values ('30', '3', '80');
insert into Exam (exam_id, student_id, score) values ('30', '4', '90');
insert into Exam (exam_id, student_id, score) values ('40', '1', '60');
insert into Exam (exam_id, student_id, score) values ('40', '2', '70');
insert into Exam (exam_id, student_id, score) values ('40', '4', '80');



select student_id,student_name from Student where student_id in 
(
	select x.student_id
    -- S.student_name count(x.student_id) as cnt, sum(x.is_quiet) as sum_

	from 
	(
		select t.exam_id, u.student_id, 
		case 
		when u.score < t.max_exam_score and u.score > t.min_exam_score then 1
		else 0
		end as is_quiet

		from 
		(
		select distinct exam_id, max(score) over w as max_exam_score, min(score) over w as min_exam_score
		from Exam
		window w as (partition by exam_id) 
		) t

		inner join 

		Exam u
		on t.exam_id = u.exam_id
	)x
	group by x.student_id
	having count(x.student_id) = sum(x.is_quiet)


);



