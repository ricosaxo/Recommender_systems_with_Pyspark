import os
import re
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, count, desc

# Initialize Spark Session
def initialize_spark():
    spark = SparkSession.builder \
        .appName('Recommender') \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.python.worker.timeout", "300") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")  # Suppress warnings
    return spark

# Load the ratings and movies data
def load_data(spark):
    ratings = spark.read.csv('ratings.csv', inferSchema=True, header=True)
    movies = spark.read.csv('movies.csv', inferSchema=True, header=True)
    return ratings, movies

# Train the ALS model
def train_model(_ratings):
    (training_data, _) = _ratings.randomSplit([0.8, 0.2], seed=42)
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=10, maxIter=15, regParam=0.1,
              coldStartStrategy="drop", nonnegative=True, implicitPrefs=False)
    model = als.fit(training_data)
    return model

# Generate and save recommendations for all user-movie pairs
def save_all_recommendations(model, movies, ratings, output_file="all_recommendations.csv"):
    # Create a DataFrame of all user-movie pairs
    users = ratings.select("userId").distinct()
    all_movies = movies.select("movieId").distinct()
    user_movie_pairs = users.crossJoin(all_movies)

    # Generate predicted ratings for all user-movie pairs
    predictions = model.transform(user_movie_pairs)

    # Join with movies data to get movie titles and genres
    predictions = predictions.join(movies, "movieId", "left").select("userId", "movieId", "title", "genres", "prediction")

    # Save predictions to a CSV file
    predictions.toPandas().to_csv(output_file, index=False)
    print(f"All recommendations saved to {output_file}")

# Load recommendations from a local file
def load_recommendations(file_path="all_recommendations.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Recommendations file not found at {file_path}. Generating recommendations...")
        return None

# Extract year from the title
def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return int(match.group(1))
    return None

# Get the most popular movies by genre
def get_popular_movies_by_genre(ratings, movies, genre):
    # Filter movies by genre
    genre_movies = movies.filter(col("genres").contains(genre))

    # Join with ratings to get the number of ratings per movie
    popular_movies = ratings.join(genre_movies, "movieId") \
        .groupBy("movieId", "title", "genres") \
        .agg(count("rating").alias("num_ratings")) \
        .orderBy(desc("num_ratings"))

    return popular_movies

# Streamlit App
def main():
    st.title("Movie Recommendation System")
    st.write("Enter your User ID and optionally choose a genre and decade!")

    # Check if recommendations file exists
    recommendations_file = "all_recommendations.csv"
    recommendations = load_recommendations(recommendations_file)

    # If recommendations file doesn't exist, generate and save recommendations
    if recommendations is None:
        spark = initialize_spark()
        ratings, movies = load_data(spark)
        model = train_model(ratings)
        save_all_recommendations(model, movies, ratings, output_file=recommendations_file)
        recommendations = load_recommendations(recommendations_file)
        spark.stop()

    if not recommendations.empty:
        # Input for user ID
        user_id = st.number_input("Enter your User ID", min_value=1, max_value=recommendations["userId"].max(), value=1)

        # Input for the number of recommendations
        num_recommendations = st.number_input("How many recommendations do you want?", min_value=1, max_value=20, value=5)

        # Get a list of unique genres
        all_genres = recommendations["genres"].str.split("|", expand=True).stack().unique()
        all_genres = sorted(all_genres)

        # Add "None" option for mixed recommendations
        all_genres.insert(0, "None")

        # Input for genre selection
        selected_genre = st.selectbox("Choose a genre (optional)", all_genres)

        # Extract year from the title and calculate decade
        recommendations["year"] = recommendations["title"].apply(extract_year)
        recommendations["decade"] = (recommendations["year"] // 10) * 10

        # Get a list of unique decades
        all_decades = sorted(recommendations["decade"].dropna().unique().astype(int))
        all_decades.insert(0, "None")  # Add "None" option for no decade filter

        # Input for decade selection
        selected_decade = st.selectbox("Choose a decade (optional)", all_decades)

        # Filter recommendations for the selected user, genre, and decade
        user_recs = recommendations[recommendations["userId"] == user_id]

        if selected_genre != "None":
            user_recs = user_recs[user_recs["genres"].str.contains(selected_genre, regex=False)]

        if selected_decade != "None":
            user_recs = user_recs[user_recs["decade"] == selected_decade]

        # Sort by predicted rating in descending order and take the top n recommendations
        user_recs = user_recs.sort_values("prediction", ascending=False).head(num_recommendations)

        # If there are fewer recommendations than requested, fill with popular movies in the genre and decade
        if len(user_recs) < num_recommendations and (selected_genre != "None" or selected_decade != "None"):
            spark = initialize_spark()
            ratings_spark, movies_spark = load_data(spark)

            # Filter movies by genre and decade
            if selected_genre != "None":
                movies_spark = movies_spark.filter(col("genres").contains(selected_genre))
            if selected_decade != "None":
                movies_spark = movies_spark.withColumn("year", col("title").rlike(r"\((\d{4})\)"))
                movies_spark = movies_spark.withColumn("decade", (col("year") // 10) * 10)
                movies_spark = movies_spark.filter(col("decade") == selected_decade)

            popular_movies = get_popular_movies_by_genre(ratings_spark, movies_spark, selected_genre)
            popular_movies = popular_movies.toPandas()

            # Exclude movies already recommended
            popular_movies = popular_movies[~popular_movies["movieId"].isin(user_recs["movieId"])]

            # Fill the remaining slots with popular movies
            remaining_slots = num_recommendations - len(user_recs)
            fill_movies = popular_movies.head(remaining_slots)
            user_recs = pd.concat([user_recs, fill_movies], ignore_index=True)
            spark.stop()

        # Display the recommendations
        if user_recs.empty:
            st.write("No recommendations available for the selected user, genre, and decade.")
        else:
            st.write(f"Top {num_recommendations} movie recommendations for User ID {user_id}:")
            st.dataframe(user_recs[["title", "genres", "prediction"]], hide_index=True)

if __name__ == "__main__":
    main()
    