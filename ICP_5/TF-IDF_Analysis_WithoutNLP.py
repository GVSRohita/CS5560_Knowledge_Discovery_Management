# Import libraries
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F

# Create a spark session to initiate the process
spark = SparkSession.builder.appName("TfIdf-tokenizing").getOrCreate()

# Read input from 5 text sources using a spark data frame
documents = spark.read.text("dataset/*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))
documents.printSchema()

# Tokens identified and extracted from the input data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)

# Compute term-frequency (tf) for the tokens identified
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Compute Inverse Document Frequency (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Display results obtained
rescaledData.select("doc_id", "features").show(truncate=False)

# Close the spark session
spark.stop()