#https://leetcode.com/problems/tournament-winners
#https://leetcode.jp/problemdetail.php?id=1194

use amazon;
SET SQL_SAFE_UPDATES = 0;

Create table If Not Exists Players (player_id int, group_id int);
Create table If Not Exists Matches (match_id int, first_player int, second_player int, first_score int, second_score int);
Truncate table Players;
insert into Players (player_id, group_id) values ('10', '2');
insert into Players (player_id, group_id) values ('15', '1');
insert into Players (player_id, group_id) values ('20', '3');
insert into Players (player_id, group_id) values ('25', '1');
insert into Players (player_id, group_id) values ('30', '1');
insert into Players (player_id, group_id) values ('35', '2');
insert into Players (player_id, group_id) values ('40', '3');
insert into Players (player_id, group_id) values ('45', '1');
insert into Players (player_id, group_id) values ('50', '2');
Truncate table Matches;
insert into Matches (match_id, first_player, second_player, first_score, second_score) values ('1', '15', '45', '3', '0');
insert into Matches (match_id, first_player, second_player, first_score, second_score) values ('2', '30', '25', '1', '2');
insert into Matches (match_id, first_player, second_player, first_score, second_score) values ('3', '30', '15', '2', '0');
insert into Matches (match_id, first_player, second_player, first_score, second_score) values ('4', '40', '20', '5', '2');
insert into Matches (match_id, first_player, second_player, first_score, second_score) values ('5', '35', '50', '1', '1');

Show tables;

select group_id, player_id from
(
	select player_id, group_id, total_score, dense_rank() over w as rank_
	from
	(
		select player_id, group_id, sum(player_score) as total_score
		from
		(
			select t1.player_id, t1.group_id,
			case when t2.first_player = t1.player_id then t2.first_score
				 when t2.second_player = t1.player_id then t2.second_score
				 else 0
			end as player_score

			from Players t1
			left join 
			Matches t2
			on t1.player_id = t2.first_player or t1.player_id = t2.second_player
		) x
		group by group_id, player_id
		order by group_id, player_id
	) y
	window w as (partition by group_id order by total_score desc, player_id)
)z 
where rank_=1;
