from flask import Flask, request, jsonify, render_template
import joblib
import nltk
nltk.data.path.append("nltk_data")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
engstopwords = stopwords.words("english")
import string
import numpy
# Download NLTK resources (only first time)

# Initialize Flask app
application = Flask(__name__)

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
model = joblib.load("Chatmodel.joblib")           # change filename if needed
vectorizer = joblib.load("chatbotVectorizer.joblib")
responses = joblib.load("chatresponses.joblib")
labels = joblib.load("labels.joblib") # change filename if needed

# ---------------------------
# NLP Preprocessing
# ---------------------------
stop_words = set(engstopwords)

def preprocess(text):
    tokens = word_tokenize(text.lower())
    cleanedTokens = [t for t in tokens if t not in engstopwords and t not in string.punctuation]
    return " ".join(cleanedTokens)

@application.route("/")
def home():
    return render_template("index.html")
# ---------------------------
# Chatbot Prediction Route
# ---------------------------
@application.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if "message" not in data:
        return jsonify({"error": "Provide 'message' field"}), 400

    user_msg = data["message"]

    # Preprocess message
    clean_msg = [preprocess(user_msg)]

    # Vectorize
    X = vectorizer.transform(clean_msg)

    # Predict label
    pred = model.predict(X)
    intent = numpy.array(labels)[pred[0]]

    response = numpy.random.choice(responses[intent])

    return jsonify({
        "user_message": user_msg,
        "prediction": response
    })


# ---------------------------
# Root
# ---------------------------


# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    application.run()
