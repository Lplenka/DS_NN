#https://leetcode.jp/problemdetail.php?id=1384
use amazon;
SET SQL_SAFE_UPDATES = 0;
Create table If Not Exists Product3 (product_id int, product_name varchar(30));
Create table If Not Exists Sales3 (product_id varchar(30), period_start date, period_end date, average_daily_sales int);
Truncate table Product3;
insert into Product3 (product_id, product_name) values ('1', 'LC Phone ');
insert into Product3 (product_id, product_name) values ('2', 'LC T-Shirt');
insert into Product3 (product_id, product_name) values ('3', 'LC Keychain');
Truncate table Sales3;
insert into Sales3 (product_id, period_start, period_end, average_daily_sales) values ('1', '2019-01-25', '2019-02-28', '100');
insert into Sales3 (product_id, period_start, period_end, average_daily_sales) values ('2', '2018-12-01', '2020-01-01', '10');
insert into Sales3 (product_id, period_start, period_end, average_daily_sales) values ('3', '2019-12-01', '2020-01-31', '1');


select period_start, period_end,
(datediff(CASE WHEN period_end   > '2018-12-31' THEN '2018-12-31' ELSE period_end end, 
	CASE WHEN period_start < '2018-01-01' THEN '2018-01-01' ELSE period_start END) + 1) * average_daily_sales,
 "2018" as report_year
from Sales3 
UNION ALL
select period_start, period_end,
(datediff( CASE WHEN period_end   > '2019-12-31' THEN '2019-12-31' ELSE period_end  END, 
CASE WHEN period_start < '2019-01-01' THEN '2019-01-01' ELSE period_start END) + 1) * average_daily_sales,
 "2019" as report_year
from Sales3
UNION ALL
select period_start, period_end,
(datediff( CASE WHEN period_end   > '2020-12-31' THEN '2020-12-31' ELSE period_end END,
CASE WHEN period_start < '2020-01-01' THEN '2020-01-01' ELSE period_start END) + 1) * average_daily_sales,
 "2020" as report_year
from Sales3;

select period_start, period_end, CASE WHEN period_end   > '2020-12-31' THEN '2020-12-31' ELSE period_end END as x,
CASE WHEN period_start < '2020-01-01' THEN '2020-01-01' ELSE period_start END as y ,
 "2020" as report_year
from Sales3;


#solution
# Time:  O(nlogn)
# Space: O(n)

SELECT product_id, 
       product_name, 
       report_year, 
       (DATEDIFF( 
           CASE WHEN YEAR(period_end)   > report_year THEN CONCAT(report_year, '-12-31') ELSE period_end   END,
           CASE WHEN YEAR(period_start) < report_year THEN CONCAT(report_year, '-01-01') ELSE period_start END
        ) + 1) * average_daily_sales AS total_amount
FROM   (SELECT s.product_id,
               product_name,
               period_start,
               period_end,
               average_daily_sales
        FROM  Sales3 s
        INNER JOIN Product3 p
        ON s.product_id = p.product_id
       ) AS r,
       (SELECT "2018" AS report_year 
        UNION ALL 
        SELECT "2019" 
        UNION ALL 
        SELECT "2020"
       ) AS y
WHERE  YEAR(period_start) <= report_year AND 
       YEAR(period_end)   >= report_year
GROUP  BY product_id,
          report_year
ORDER  BY product_id,
          report_year;
           
           
# Time:  O(nlogn)
# Space: O(n)
SELECT r.product_id, 
       product_name, 
       report_year, 
       total_amount 
FROM   ((SELECT product_id, 
                '2018'                     AS report_year, 
                days * average_daily_sales AS total_amount 
         FROM   (SELECT product_id, 
                        average_daily_sales, 
                        DATEDIFF(
                             CASE WHEN period_end   > '2018-12-31' THEN '2018-12-31' ELSE period_end  END,
                             CASE WHEN period_start < '2018-01-01' THEN '2018-01-01' ELSE period_start END
                        ) + 1 AS days 
                 FROM   Sales3 s) tmp 
         WHERE  days > 0) 
        UNION ALL
        (SELECT product_id, 
                '2019'                     AS report_year, 
                days * average_daily_sales AS total_amount 
         FROM   (SELECT product_id, 
                        average_daily_sales, 
                        DATEDIFF(
                             CASE WHEN period_end   > '2019-12-31' THEN '2019-12-31' ELSE period_end  END,
                             CASE WHEN period_start < '2019-01-01' THEN '2019-01-01' ELSE period_start END
                        ) + 1 AS days 
                 FROM   Sales3 s) tmp 
         WHERE  days > 0) 
        UNION ALL
        (SELECT product_id, 
                '2020'                     AS report_year, 
                days * average_daily_sales AS total_amount 
         FROM   (SELECT product_id, 
                        average_daily_sales, 
                        DATEDIFF(
                             CASE WHEN period_end   > '2020-12-31' THEN '2020-12-31' ELSE period_end END,
                             CASE WHEN period_start < '2020-01-01' THEN '2020-01-01' ELSE period_start END
                        ) + 1 AS days 
                 FROM   Sales3 s) tmp 
         WHERE  days > 0)
       ) r
       INNER JOIN Product3 p
      ON r.product_id = p.product_id
ORDER  BY r.product_id, 
          report_year ;