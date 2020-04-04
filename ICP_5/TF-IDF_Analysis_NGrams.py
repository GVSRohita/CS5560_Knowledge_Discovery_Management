# Import library
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

# Create a spark session to initiate process
spark = SparkSession.builder.appName("TfIdf-Ngram").getOrCreate()

# Read input from 5 text sources using a spark data frame
documents = spark.read.text("dataset/*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))
documents.printSchema()

# Tokens identified and extracted from input data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)

# Identify n-grams (3-word combinations) from the tokens derived
ngram = NGram(n=3, inputCol="words", outputCol="ngrams")
ngramDataFrame = ngram.transform(wordsData)

# Compute term-frequency (TF) for the tokens identified
hashingTF = HashingTF(inputCol="ngrams", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(ngramDataFrame)

# Compute inverse document frequency (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Display results obtained
rescaledData.select("doc_id", "features").show(truncate=False)

# Close the spark session
spark.stop()