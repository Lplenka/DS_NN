Please open the notebooks in jupyter or open the mhtml files in chrome/Microsoft_Edge

* Question 1 : Solution Notebook [shopify_analysis](https://github.com/Lplenka/DS_NN/blob/master/Lalu_Lenka_Shopify_Submission/shopify_analysis.ipynb)

   * It was observed that the Average Order Value of '*3145.13*' was abnormally high. We observed that the distribution on order_amount was extremely right skewed which is again abnormal.

    * I analysed the data, specifically columns order_amount, total_items and shop_ids. 

    * It was observed that shops with id 42 has abnormally high order amount values for customer with id 607 who has ordered 2000 items per order multiple times. Shop with id 78 has all its orders extremely high compared to majority of the shops.

    #### Suggested Metrics

    * Use Median Order Value (MOV) as a metric as it works well for skewed data, this way we don't have to remove outliers. The Median Order value for given data is *284*.

    * Remove outlier shop id 42 and 78 and then calculate the Average Order Value (AOV). The Average order value after removing outlier comes around *300.16*.



* Question 2 : Solution Notebook [SQL Questions](https://github.com/Lplenka/DS_NN/blob/master/Lalu_Lenka_Shopify_Submission/SQL_Questions.ipynb)

    * How many orders were shipped by Speedy Express in total?
        - Answer : 54

    * What is the last name of the employee with the most orders?
        - Answer: Peacock

    * What product was ordered the most by customers in Germany?
        - Answer: Boston Crab Meat


