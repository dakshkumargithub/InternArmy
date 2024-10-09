import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import joblib

# Download stopwords once (if not already downloaded)
nltk.download('stopwords', quiet=True)

# Load the dataset
news_df = pd.read_csv('train.csv')

# Fill missing values
news_df = news_df.fillna(' ')

# Create a new content column by combining author and title
news_df['content'] = news_df['author'] + " " + news_df['title']

# Stemming function
ps = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)


# Apply stemming
news_df['content'] = news_df['content'].apply(stemming)

# Prepare features and labels
X = news_df['content'].values
y = news_df['label'].values

# Vectorization
vector = TfidfVectorizer()
X = vector.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_y_pred = model.predict(X_train)
print("Train accuracy:", accuracy_score(train_y_pred, y_train))

test_y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(test_y_pred, y_test))

# Save the model and vectorizer for future use
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vector, 'tfidf_vectorizer.pkl')


# Prediction function
def predict(input_data):
    input_data = [input_data]  # Wrap in list for single sample prediction
    input_vectorized = vector.transform(input_data)
    prediction = model.predict(input_vectorized)

    return "Fake news" if prediction[0] == 1 else "Real news"


# User input for prediction
user_input = input("Enter the news article content: ")
result = predict(user_input)

# Display prediction result
print("Prediction:", result)
