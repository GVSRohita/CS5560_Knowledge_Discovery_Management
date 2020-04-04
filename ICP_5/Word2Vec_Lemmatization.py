# Import library
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from nltk.stem import WordNetLemmatizer
from pyspark.sql.types import ArrayType, StringType
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.feature import Word2Vec

# Define lemmatization process
lemmtizer = WordNetLemmatizer()
def lemmetize(input_list):
    print(input_list)
    if(len(input_list) == 1):
        return list()
    return [lemmtizer.lemmatize(word) for word in input_list]

# Create spark session to initiate process
spark = SparkSession.builder.appName("TfIdf-Lemmetization").getOrCreate()

# Initiate lemmatization process
lemmetize = F.udf(lemmetize)

# Read input from 5 text sources using a spark data frame
documents = spark.read.text("dataset/*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))
documents.printSchema()

# Toekns identified and extracted from input data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)
wordsData.show()

# Stem words derived using the tokens identified
stemmer = SnowballStemmer(language='english')
stemmer_udf = F.udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
wordsData = wordsData.withColumn("lemms", stemmer_udf("words"))

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="lemms", outputCol="result")
model = word2Vec.fit(wordsData)
result = model.transform(wordsData)

# Display synonyms and cosine similarity of words within input data
synonyms = model.findSynonyms("5g", 5)  # its okay for certain words , real bad for others
synonyms.show(5)

# Close the spark session
spark.stop()