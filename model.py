import pandas as pd
# Load the new dataset
df_new = pd.read_csv("D:\ASOIU MASTER\II kurs II sem\Fuzzy\midterm\movies\movies1.csv")

# Display basic info and preview
df_new.info(), df_new.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Drop rows with missing target (just in case)
df_model = df_new.dropna(subset=['Gross'])

# Define features and target
X = df_model[['Budget', 'MPAA Rating', 'Genre', 'Runtime', 'Rating', 'Rating Count']]
y = df_model['Gross']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical and numerical columns
categorical_cols = ['MPAA Rating', 'Genre']
numeric_cols = ['Budget', 'Runtime', 'Rating', 'Rating Count']

# # Preprocessing
# preprocessor = ColumnTransformer([
#     ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#     ('num', 'passthrough', numeric_cols)
# ])

from sklearn.impute import SimpleImputer

# Updated preprocessor with imputers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])

# Final pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Save the trained pipeline
model_path = "D:\ASOIU MASTER\II kurs II sem\Fuzzy\midterm\movie_gross_predictor.pkl"
joblib.dump(pipeline, model_path)

mae, model_path

