from flask import Flask, request, render_template, jsonify
import joblib
import re
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import firebase_admin
from firebase_admin import credentials, firestore

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("authenticatorapp-10fc8-firebase-adminsdk-m7efw-bb6797ccd9.json")
firebase_admin.initialize_app(cred)
db = firestore.client()



model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer (1).pkl')

# Define text preprocessing functions
def cleanstr(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test_static')
def test_static():
    return app.send_static_file('styles.css')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    name = data.get('name', 'Anonymous')
    comment_text = data.get('comment', '')

    # Preprocess the input text
    cleaned_text = cleanstr(comment_text)
    cleaned_text = remove_stopwords(cleaned_text)
    cleaned_text = lemmatize(cleaned_text)

    # Transform the text using the TF-IDF vectorizer
    text_vector = tfidf.transform([cleaned_text])

    # Predict the sentiment using the loaded model
    prediction = model.predict(text_vector)[0]

    # Determine the rating based on sentiment
    if prediction == 1:
        prediction_str = 'Positive'
        rating = 5
    else:
        prediction_str = 'Negative'
        rating = 1

    # Save the comment, sentiment, rating, and name to Firebase Firestore
    doc_ref = db.collection('comments').add({
        'name': name,
        'comment': comment_text,
        'sentiment': prediction_str,
        'rating': rating,
        'timestamp': datetime.datetime.utcnow()
    })
    print("Name:", name)  # Debug statement
    print("Prediction:", prediction_str, "Rating:", rating)  # Debug statement
    return jsonify({'prediction': prediction_str, 'rating': rating})

@app.route('/get_comments', methods=['GET'])
def get_comments():
    comments_ref = db.collection('comments').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
    comments = []
    for comment in comments_ref:
        comment_data = comment.to_dict()
        comments.append({
            'name': comment_data.get('name', 'Anonymous'),
            'comment': comment_data.get('comment', ''),
            'sentiment': comment_data.get('sentiment', 'Unknown'),
            'rating': comment_data.get('rating', 0),
            'timestamp': comment_data.get('timestamp', datetime.datetime.utcnow())
        })
    return jsonify(comments)

@app.route('/get_results', methods=['GET'])
def get_results():
    comments_ref = db.collection('comments').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
    positive_comments = []
    negative_comments = []
    total_comments = 0
    for comment in comments_ref:
        comment_data = comment.to_dict()
        total_comments += 1
        if comment_data.get('sentiment') == 'Positive':
            positive_comments.append(comment_data.get('comment'))
        else:
            negative_comments.append(comment_data.get('comment'))
    
    positive_percentage = (len(positive_comments) / total_comments) * 100 if total_comments > 0 else 0
    negative_percentage = (len(negative_comments) / total_comments) * 100 if total_comments > 0 else 0
    
    return jsonify({
        'positive_comments': positive_comments,
        'negative_comments': negative_comments,
        'positive_percentage': positive_percentage,
        'negative_percentage': negative_percentage
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


    '''print("Prediction:", prediction_str, "Rating:", rating)  # Debug statement'''