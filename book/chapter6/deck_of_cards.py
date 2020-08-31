import pandas as pd

df = pd.DataFrame(
[
(1, "202004", 10, 'indihome'),
(1, "202004", 20, 'indihome'),
(2, "202004", 40, 'myrepublic'),
(2, "202005", 5, 'myrepublic'),
(2, "202005", 100, 'xlhome'),
(2, "202005", 20, 'xlhome'),
(2, "202005", 40, 'xlhome'),
(2, "202005", 80 , 'myrepublicindonesia'),
(4, "202005", 90, 'myrepublicindonesia'),
(4, "202005", 10, 'abc'),
(5, "202005", 80, 'abc'),
(5, "202004", 50, 'abc'),
(4, "202004", 10, 'indihome')
]
,
columns = ["msisdn", "month", "trx", "apps"]

)

print(df)

df_ = df.groupby("msisdn", "month").agg(f.count(f.when(f.col("apps").isin(),"trx")))

#.otherwise(0)).cast(t.LongType()).alias("sum"))

print(df_)
