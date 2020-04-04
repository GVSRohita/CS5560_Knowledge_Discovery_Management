# Import library
from pyspark.ml.feature import Tokenizer, NGram
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec

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
ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
wordsData = ngram.transform(wordsData)

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="ngrams", outputCol="result")
model = word2Vec.fit(wordsData)
result = model.transform(wordsData)

# Display synonyms and cosine similarity of words within input data
synonyms = model.findSynonyms("5g", 5)  # its okay for certain words , real bad for others
synonyms.show(5)

# Close the spark session
spark.stop()