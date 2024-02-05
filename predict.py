import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Load the pre-trained model and vectorizer
loaded_model = pickle.load(open('E:/twitter/trained_model.pkl', 'rb'))
loaded_vectorizer = pickle.load(open('E:/twitter/vectorizer.pkl', 'rb'))

# Define Porter Stemmer
port_stem = PorterStemmer()

def preprocess_tweet(tweet, port_stem):

    stemmed_content = re.sub('[^a-zA-Z]',' ',tweet)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

# Input a tweet for prediction
input_tweet = input("Enter a tweet for prediction: ")

# Preprocess the input tweet
processed_tweet = preprocess_tweet(input_tweet, port_stem)

# Vectorize the tweet using the pre-trained vectorizer
input_data = loaded_vectorizer.transform([processed_tweet])

# Make prediction using the loaded model
prediction = loaded_model.predict(input_data)

# Determine the sentiment
sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

# Print the result
print(f"Input Tweet: {input_tweet}")
print(f"Predicted Sentiment: {sentiment}")