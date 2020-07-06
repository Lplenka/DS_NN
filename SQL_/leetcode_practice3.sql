#https://leetcode.com/problems/trips-and-users/

use amazon;
SET SQL_SAFE_UPDATES = 0;
drop table Trips;
create table Trips
(
Id int not null auto_increment,
Client_Id int,
Driver_Id int,
City_Id int,
Status varchar(225),
Request_at DATE,

PRIMARY KEY (id)	

);
delete from Trips;

insert into Trips (Client_Id, Driver_Id, City_Id, Status, Request_at)
values
(1, 10, 1, "completed" ,"2013-10-01"),
(2, 11, 1, "cancelled_by_driver" ,"2013-10-01"),
(3, 12, 6, "completed" ,"2013-10-01"),
(4, 13, 6, "cancelled_by_client" ,"2013-10-01"),
(1, 10, 1, "completed" ,"2013-10-02"),
(2, 11, 6, "completed" ,"2013-10-02"),
(3, 12, 6, "completed" ,"2013-10-02"),
(2, 12, 12, "completed" ,"2013-10-03"),
(3, 10, 12, "completed" ,"2013-10-03"),
(4, 13, 12, "cancelled_by_driver" ,"2013-10-03");

drop table Users;
create table Users
(
Users_Id int,
Banned varchar(225),
Role varchar(225)
);


delete from Users;
insert into Users (Users_Id, Banned, Role)
values
(1,'No', "client"),
(2,'Yes',"client"),
(3,'No', "client"),
(4,'No', "client"),
(10,'No', "driver"),
(11,'No', "driver"),
(12,'No',"driver"),
(13,'No', "driver");

select * from Trips;
select * from Users;

select  P.Request_at as Day, round(count(if(P.Status like "%cancelled%", 1, null))/count(P.Status),2) as "Cancellation Rate" 
from
(
select T.Client_Id, T.Driver_Id, T.Status, T.Request_at, U.Users_Id from Trips T
join 
(select Users_Id from Users where Banned = "No") U
where T.Client_Id = U.Users_Id and T.Driver_Id in (select Users_Id from Users where Banned = "No") 
and T.Request_at between "2013-10-01" and "2013-10-03"

) P
group by  P.Request_at;