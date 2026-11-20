import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/messages.csv")

# Clean dataset
data["message"] = data["message"].astype(str)
data["label"] = data["label"].astype(str).str.lower().str.strip()

data = data[data["label"].isin(["spam", "ham"])]

# Separate data
X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)

# Test model
predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("\nEMAIL SPAM DETECTION SYSTEM")
print("---------------------------")

print("Total Messages:", len(data))
print("Training Messages:", len(X_train))
print("Testing Messages:", len(X_test))

print("Model Training Completed!")

print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))