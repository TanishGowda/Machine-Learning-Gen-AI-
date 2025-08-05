from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Create a Spark session
spark = SparkSession.builder.appName("SpamClassifier").getOrCreate()

data = [
    (0, "Hi, how are you?"),
    (1, "Win money now!!! Click here"),
    (0, "Let's meet for lunch"),
    (1, "Congratulations, you won a lottery")
]

df = spark.createDataFrame(data, ["label", "message"])

# Pipeline Stages
tokenizer = Tokenizer(inputCol="message", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)

# Build pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

# Training the model
model = pipeline.fit(df)

test = spark.createDataFrame([(0, "Free money offer just for you")], ["label", "message"])
prediction = model.transform(test)
prediction.select("message", "probability", "prediction").show()

spark.stop()
