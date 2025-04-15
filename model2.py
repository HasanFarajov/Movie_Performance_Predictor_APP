# Re-import everything after kernel reset for model training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load base dataset
movies_df = pd.read_csv("D:\ASOIU MASTER\II kurs II sem\Fuzzy\midterm\movies\movies1.csv")
actors_df = pd.read_csv("D:\ASOIU MASTER\II kurs II sem\Fuzzy\midterm\movies\\actors.csv")
movies_actors_df = pd.read_csv("D:\ASOIU MASTER\II kurs II sem\Fuzzy\midterm\movies\moviesactors.csv")

# Merge actors with movie_actor link
merged_df = movies_actors_df.merge(actors_df, on="ActorID", how="left")
actor_features = merged_df.groupby("MovieID").agg({
    "ActorID": "count",
    "Height (Inches)": "mean",
    "Gender": lambda x: (x == "Male").sum() / x.count() if x.count() > 0 else 0
}).rename(columns={
    "ActorID": "Actor Count",
    "Height (Inches)": "Avg Height",
    "Gender": "Male Ratio"
}).reset_index()

# Merge actor features with main dataset
df = movies_df.merge(actor_features, on="MovieID", how="left")

# Drop rows with missing target and keep relevant columns
df = df.dropna(subset=["Gross"])

# Define features and target
X = df[[
    "Budget", "MPAA Rating", "Genre", "Runtime", "Rating", "Rating Count",
    "Actor Count", "Avg Height", "Male Ratio"
]]
y = df["Gross"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define transformers
numeric_cols = ["Budget", "Runtime", "Rating", "Rating Count", "Actor Count", "Avg Height", "Male Ratio"]
categorical_cols = ["MPAA Rating", "Genre"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full preprocessor
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Save updated model
model_path = "D:\ASOIU MASTER\II kurs II sem\Fuzzy\midterm\movie_gross_predictor_v2.pkl"
joblib.dump(pipeline, model_path)

model_path
