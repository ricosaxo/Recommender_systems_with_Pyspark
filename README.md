# Movie Recommendation System

This project is a movie recommendation system built using PySpark and Streamlit. It leverages the Alternating Least Squares (ALS) algorithm to provide personalized movie recommendations based on user preferences. The system allows users to input their User ID and optionally filter recommendations by genre and decade.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Model](#model)
- [Deliverables](#deliverables)
- [Contributors](#contributors)
- [Timeline](#timeline)

## Description

The Movie Recommendation System is designed to provide personalized movie recommendations to users based on their historical ratings. The system uses collaborative filtering with the ALS algorithm to predict user preferences and recommend movies. The front-end interface is built using Streamlit, making it user-friendly and interactive.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

```bash
   git clone https://github.com/your-username/Recommender-systems-with-Pyspark.git
   cd Recommender-systems-with-Pyspark
```

2. **Install the required dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download the dataset**:

Download the MovieLens dataset from Kaggle.

Place the ratings.csv and movies.csv files in the project directory.

4. **Run the recommendation system**:

```bash
streamlit run app.py
```

5. **Usage**:

Launch the Streamlit app:

```bash
streamlit run app2.py
```

6. **Enter a User ID**:

Input your a user ID in the provided field.

7. **Filter recommendations (optional)**:

Choose a genre and/or decade to filter the recommendations.

8. **View recommendations**:

The system will display the top movie recommendations based on your preferences.

## Data Sources

MovieLens Dataset: Kaggle

IMDB Dataset: IMDB

## Model

The recommendation system uses the Alternating Least Squares (ALS) algorithm, a collaborative filtering technique implemented in PySpark's MLlib. The model is trained on user ratings to predict preferences and generate recommendations.

Key Steps:

Data Preprocessing: Load and preprocess the ratings and movies data.

Model Training: Train the ALS model on the training dataset.

Prediction: Generate recommendations for all users and save them to a CSV file.

Evaluation: Evaluate the model using Root Mean Squared Error (RMSE).

## Deliverables

Source Code: Published on GitHub.

README: Complete and well-documented.

Presentation: 10-minute presentation of results with 5 minutes Q&A.

Contributors: List of contributors.

Timeline: Project timeline and milestones.

## Contributors

Rik Sas

## Timeline

Day 1: Project setup and data collection.

Day 2: Model development and training.

Day 3: Streamlit app development and deployment.
