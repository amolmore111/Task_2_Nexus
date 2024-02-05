from flask import Flask, render_template, request, jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model and vectorizer using relative paths
model_path = os.path.join(current_dir, 'trained_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

loaded_model = pickle.load(open(model_path, 'rb'))
loaded_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Define Porter Stemmer
port_stem = PorterStemmer()

def preprocess_tweet(tweet, port_stem):
    # Perform the preprocessing steps 
    stemmed_content = re.sub('[^a-zA-Z]',' ',tweet)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            tweet = request.form['tweet']

            # Preprocess the input tweet
            processed_tweet = preprocess_tweet(tweet, port_stem)

            # Vectorize the tweet using the pre-trained vectorizer
            input_data = loaded_vectorizer.transform([processed_tweet])

            # Make prediction using the loaded model
            prediction = loaded_model.predict(input_data)

            # Determine the sentiment
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

            return render_template('result.html', tweet=tweet, sentiment=sentiment)

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
