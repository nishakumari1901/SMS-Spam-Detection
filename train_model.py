import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
data = pd.read_csv("dataset/sms-spam.csv")

# Rename columns for clarity
data = data.rename(columns={'v1': 'label', 'v2': 'message'})

# Map 'ham' to 0 and 'spam' to 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and labels
X = data['message']
y = data['label']

# Convert text data to numerical using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer as pickle files
with open('sms_spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('sms_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("Model training complete and saved as 'sms_spam_model.pkl'!")
