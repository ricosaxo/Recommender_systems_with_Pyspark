# Importing the required pyspark library
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import avg, min, max
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Setup Spark Session
spark = SparkSession.builder.appName('Recommender').getOrCreate()
spark

# Read the ratings data
ratings = spark.read.csv('ratings.csv', inferSchema=True, header=True)
ratings.show(5)

# Count the total number of ratings in the dataset
numerator = ratings.select("rating").count()

# Count the number of distinct userIds and distinct movieIds
num_users = ratings.select("userId").distinct().count()
num_movies = ratings.select("movieId").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator * 1.0) / denominator) * 100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")

# Min num ratings for movies
print("Movie with the fewest ratings: ")
ratings.groupBy("movieId").count().select(min("count")).show()

# Avg num ratings per movie
print("Avg num ratings per movie: ")
ratings.groupBy("movieId").count().select(avg("count")).show()

# Print the schema of the ratings dataframe
ratings.printSchema()

# Split the data into training and test sets
(training_data, test_data) = ratings.randomSplit([0.8, 0.2], seed=42)

# Initialize the ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=10, maxIter=15, regParam=0.1,
          coldStartStrategy="drop", nonnegative=True, implicitPrefs=False)

# Fit the model on the training data
model = als.fit(training_data)

# Make predictions on the test data
test_predictions = model.transform(test_data)
test_predictions.show(10)

# Initialize the evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

print(evaluator.getMetricName())
print(evaluator.getLabelCol())
print(evaluator.getPredictionCol())

# Evaluate the model using RMSE
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)

# Generate recommendations for all users
n = 5
ALS_recommendations = model.recommendForAllUsers(n)
ALS_recommendations.show()

# Create a temporary view for the recommendations
ALS_recommendations.createOrReplaceTempView("ALS_recs_temp")

# Clean the recommendations
clean_recs = spark.sql("SELECT userId, movieIds_and_ratings.movieId AS movieId, movieIds_and_ratings.rating AS prediction FROM ALS_recs_temp LATERAL VIEW explode(recommendations) exploded_table AS movieIds_and_ratings")

# Explode the recommendations
exploded_recs = spark.sql("SELECT userId, explode(recommendations) AS MovieRec FROM ALS_recs_temp")
exploded_recs.show()

# Show the cleaned recommendations
clean_recs.show()

# Read the movie information
movie_info = spark.read.csv('movies.csv', inferSchema=True, header=True)
movie_info.show(5)

# Join the cleaned recommendations with movie information
clean_recs = clean_recs.join(movie_info, ["movieId"], "left")
clean_recs.show()

# Join the cleaned recommendations with ratings to find unrated movies
clean_recs = clean_recs.join(ratings, ["userId", "movieId"], "left").filter(ratings['rating'].isNull())
clean_recs.show()

# Uncomment the following lines if you want to perform hyperparameter tuning
# param_grid = ParamGridBuilder().addGrid(als.rank, [5, 40]).addGrid(als.maxIter, [5, 100]).addGrid(als.regParam, [.05, .1]).build()
# cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
# model = cv.fit(training_data)

