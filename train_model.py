import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text


data = pd.read_csv("IMDB Dataset.csv")

data["review"] = data["review"].apply(clean_text)

X = data["review"]
y = data["sentiment"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained successfully!")