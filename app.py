from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
data = pd.read_csv("spam.csv")

# Features and labels
X_text = data["message"]
y = data["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

print("Accuracy =", accuracy)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]

        if message.strip():
            message_vector = vectorizer.transform([message])

            result = model.predict(message_vector)[0]

            if result == "spam":
                prediction = "🚨 Spam Message"
            else:
                prediction = "✅ Safe Message"

    return render_template(
        "index.html",
        prediction=prediction,
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(debug=True)