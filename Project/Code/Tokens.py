# Import library
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from nltk.stem import WordNetLemmatizer
from pyspark.sql.types import ArrayType, StringType
from nltk.stem.snowball import SnowballStemmer

# Define Lemmatization process
lemmtizer = WordNetLemmatizer()
def lemmetize(input_list):
    print(input_list)
    if(len(input_list) == 1):
        return list()
    return [lemmtizer.lemmatize(word) for word in input_list]

# Create a spark session to initiate process
spark = SparkSession.builder.appName("TfIdf-Lemmetization").getOrCreate()

# Initiate lemmatization process
lemmetize = F.udf(lemmetize)

# Read input from 5 text sources using a spark data frame
documents = spark.read.text("dataset/*.txt")
# documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))
documents.printSchema()

# Tokens identified and extracted from input data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)
wordsData.show(truncate=False)

# Stem words derived using the tokens identified
stemmer = SnowballStemmer(language='english')
stemmer_udf = F.udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
wordsData = wordsData.withColumn("lemms", stemmer_udf("words"))
wordsData.show()

# Compute term-frequency (TF) on the tokens identified
hashingTF = HashingTF(inputCol="lemms", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Compute Inverse Document Frequency (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Display results identified
rescaledData.select("doc_id", "features").show(truncate=False)

# Close the spark session
spark.stop()