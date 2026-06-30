import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("IMDB Dataset.csv")

X = df["review"]
y = df["sentiment"]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X, y)

joblib.dump(model, "sentiment_model.pkl")

print("Model trained and saved as sentiment_model.pkl")