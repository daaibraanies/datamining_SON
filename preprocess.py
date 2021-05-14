import findspark
import pyspark
from pyspark.sql import SQLContext
import datetime
import json
import sys
import json
import pandas
findspark.find()
spark_context = pyspark.SparkContext()
sql_context = SQLContext(spark_context)

business_file = 'business.json'
review_file = 'review.json'

b_json = sql_context.read.json(business_file).rdd
allowed_ids = b_json.filter(lambda x: x['state'] == 'NV')\
              .map(lambda x:x['business_id']).collect()

r_json = sql_context.read.json(review_file).rdd
filtered_review = r_json.filter(lambda x:x['business_id'] in allowed_ids)\
                .map(lambda x:str(x['user_id'])+','+str(x['business_id'])+'\n')

with open('test.csv','w') as out:
    out.write("user_id,business_id\n")
    for line in filtered_review.collect():
        out.write(line)




