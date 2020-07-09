#https://leetcode.jp/problemdetail.php?id=1097
#https://leetcode.com/problems/game-play-analysis-v
use amazon;
SET SQL_SAFE_UPDATES = 0;
create table if not exists Activity
(
player_id int,
device_id int,
event_date date,
games_played int
);

delete from Activity;

insert into Activity ( player_id, device_id, event_date, games_played)
values
	(1,2, "2016-03-01",5),
    (1,2, "2016-03-02",6),
    (2,3, "2017-06-25",1),
    (3,1, "2016-03-01",0),
    (3,4, "2016-07-03",5);
    
    
select * from Activity;

select event_date, count(event_date) from Activity
group by event_date;

#https://code.dennyzhang.com/game-play-analysis-i
#SQL query that reports the first login date for each player.
select distinct player_id, first_value(event_date) over w as "login_date" from Activity
window w as(partition by player_id order by event_date);
#solution 2
select player_id, min(event_date) as first_login
from Activity
group by player_id;

#SQL query that reports the device that is first logged in for each player.
# https://code.dennyzhang.com/game-play-analysis-ii
select distinct player_id, device_id
from Activity
where (player_id, event_date) in (
    select player_id, min(event_date)
    from Activity
    group by player_id);

-- https://code.dennyzhang.com/game-play-analysis-iii
-- Write an SQL query that reports for each player and date, how many games played so far by the player. 
-- That is, the total number of games played by the player until that date. Check the example for clarity.

select a.player_id, a.event_date, sum(b.games_played) as games_played_so_far
from
(select player_id, event_date, games_played from Activity) a
join
(select player_id, event_date, games_played from Activity) b
on a.player_id = b.player_id
where a.event_date >= b.event_date
group by a.player_id, a.event_date
order by a.player_id, a.event_date;

-- https://code.dennyzhang.com/game-play-analysis-iv
-- Write an SQL query that reports the fraction of players that logged in again on the day after the day 
-- they first logged in, rounded to 2 decimal places. In other words, you need to count 
-- the number of players that logged in for at least two consecutive 
-- days starting from their first login date, then divide that number by the total number of players.


select t.player_id, t.first_login_date, t.second_login_date
from 
(
select player_id, 
first_value(event_date) over w as "first_login_date", 
Nth_value(event_date,2) over w as "second_login_date" 
from Activity
window w as(partition by player_id order by event_date) 
)t
where
DATEDIFF(t.second_login_date,t.first_login_date) =1;

#SOLUTION
select round(sum(case when t1.event_date = t2.first_event+1 then 1 else 0 end)/count(distinct t1.player_id), 2) as fraction
from Activity as t1 inner join
    (select player_id, min(event_date) as first_event
    from Activity
    group by player_id) as t2
on t1.player_id = t2.player_id;

#https://leetcode.jp/problemdetail.php?id=1097


select first_date as install_dt, count(t2.player_id) as installs 
from

    (select player_id, min(event_date) as first_date 
    from Activity
    group by player_id) t2

group by t2.first_date;

#############################################################################


select first_date as install_dt, count(t2.player_id) as installs, 
count(
case 
when 
EXISTS(select player_id from Activity where player_id = t2.player_id and event_date=t2.first_date + 1) 
then 1 else NULL
end
)/count(t2.player_id) as Day1_retention
from

    (select player_id, min(event_date) as first_date 
    from Activity
    group by player_id) t2

group by t2.first_date;
