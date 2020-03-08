# Import library
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

# Create a spark session
spark = SparkSession.builder.appName("TfIdf-tokenizing").getOrCreate()

# Load and read input
documents = spark.read.text("dataset/*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))
print('The schema associated to the input is')
documents.printSchema()

# Tokenization of input (splitting the text into individual token/word)
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)

# Computation of Term-Frequency (TF) associated with the data
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Computation of Inverse Document Frequency (IDF) associated with the data
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Identifying the TF_IDF associated with the data
rescaledData.select("doc_id", "features").show(truncate=False)

# Close spark session
spark.stop()