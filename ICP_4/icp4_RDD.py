import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkFiles


sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

# Reading data file in data frame in the form of RDD (Resilient Distributed Dataset)
df = sqlContext.read.csv(SparkFiles.get("C:\\Users\\Poonam\\Documents\\GitHub\\CS5560---Knowledge-Discovery-Management\\ICP_4\\data.csv"), header=True, inferSchema= True)
# print("Identifying differing schemas associated with individual data types within the columns of the data set used:")
# df.printSchema()
print("Display RDD of top 5 data elements:")
df.show(5, truncate = False)

#If you didn't set inderShema to True, here is what is happening to the type. There are all in string.
# df_string = sqlContext.read.csv(SparkFiles.get("C:\\Users\\Poonam\\Documents\\GitHub\\CS5560---Knowledge-Discovery-Management\\ICP_4\\data.csv"), header=True, inferSchema=  False)
# print("Identifying differing schemas when inferSchema is not accounted for with data types assocaited with the data set used:")
# df_string.printSchema()


#You can select and show the rows with select and the names of the features. Below, gender and churn are selected.
# print("Specific features with limited number of records displayed:")
# df.select('gender','churn').show(5)

#To get a summary statistics, of the data, you can use describe(). It will compute the :count, mean, standarddeviation, min, max
print("Statistical Summary of Data Set Features:")
df.describe().show()