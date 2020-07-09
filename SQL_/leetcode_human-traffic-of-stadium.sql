-- https://leetcode.com/problems/human-traffic-of-stadium/
# Write your MySQL query statement below

select t.id,t.visit_date, t.people from

(
    
 select id, visit_date, people, 
 LAG(people,2) over w as 'people_2',
 LAG(people,1) over w as 'people_1',
 LEAD(people,1) over w as 'people1',
 LEAD(people,2) over w as 'people2' 
 from stadium
 window w as (order by id) 

) t

where (t.people>=100
and  t.people1>=100
and t.people2>=100) 
or
(t.people>=100
and  t.people_1>=100
and t.people_2>=100)
or
(t.people_1>=100
and  t.people>=100
and t.people1>=100);