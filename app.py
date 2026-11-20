from flask import Flask, render_template, request, redirect, url_for, Response
import pandas as pd
import csv
import io

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

app = Flask(__name__)

history = []

# Load dataset
data = pd.read_csv("data/messages.csv")

# Clean dataset
data["message"] = data["message"].astype(str)
data["label"] = data["label"].astype(str).str.lower().str.strip()
data = data[data["label"].isin(["spam", "ham"])]

if data.empty:
    raise ValueError("No valid messages found in dataset.")

# Separate messages and labels
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

# Create TF-IDF vectorizer
evaluation_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vectorized = evaluation_vectorizer.fit_transform(X_train)
X_test_vectorized = evaluation_vectorizer.transform(X_test)

# Train evaluation model
evaluation_model = MultinomialNB(alpha=0.1)
evaluation_model.fit(X_train_vectorized, y_train)

# Predict test messages
predictions = evaluation_model.predict(X_test_vectorized)

# Calculate performance scores
test_accuracy = accuracy_score(y_test, predictions) * 100

precision = precision_score(
    y_test, predictions, pos_label="spam", zero_division=0
) * 100

recall = recall_score(
    y_test, predictions, pos_label="spam", zero_division=0
) * 100

f1 = f1_score(
    y_test, predictions, pos_label="spam", zero_division=0
) * 100

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

true_normal = int(cm[0][0])
false_spam = int(cm[0][1])
missed_spam = int(cm[1][0])
true_spam = int(cm[1][1])

# Train final model using complete dataset
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB(alpha=0.1)
model.fit(X_vectorized, y)

# Dataset statistics
total_messages = len(data)
spam_messages = len(data[data["label"] == "spam"])
ham_messages = len(data[data["label"] == "ham"])


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    message = ""
    status = ""
    confidence = ""

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if message:

            message_vector = vectorizer.transform([message])

            prediction = model.predict(message_vector)[0]

            probabilities = model.predict_proba(message_vector)[0]

            confidence = round(max(probabilities) * 100, 2)

            if prediction == "spam":
                result = "SPAM MESSAGE"
                status = "spam"
            else:
                result = "NOT SPAM"
                status = "ham"

            history.insert(0, {
                "message": message,
                "result": result,
                "status": status,
                "confidence": confidence
            })

            del history[10:]

    return render_template(
        "index.html",
        result=result,
        message=message,
        status=status,
        confidence=confidence,
        history=history,
        total_messages=total_messages,
        spam_messages=spam_messages,
        ham_messages=ham_messages,
        test_accuracy=round(test_accuracy, 2),
        precision=round(precision, 2),
        recall=round(recall, 2),
        f1=round(f1, 2),
        true_normal=true_normal,
        false_spam=false_spam,
        missed_spam=missed_spam,
        true_spam=true_spam
    )


@app.route("/clear")
def clear_history():

    history.clear()

    return redirect(url_for("home"))


@app.route("/download")
def download_history():

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "Message",
        "Prediction",
        "Confidence"
    ])

    for item in history:

        writer.writerow([
            item["message"],
            item["result"],
            str(item["confidence"]) + "%"
        ])

    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )

    response.headers["Content-Disposition"] = (
        "attachment; filename=spam_detection_history.csv"
    )

    return response


if __name__ == "__main__":
    app.run(debug=True)