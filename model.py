import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import joblib
import numpy as np


df = pd.read_csv('dataset2.csv')
x_train = df['Job title']

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = text.strip()
    text = text.lower()  
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

x_train = x_train.apply(clean_text)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=2, max_features=10000)
X = vectorizer.fit_transform(x_train)
y = df['Course']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
cv_results = cross_validate(model,X_train, y_train, cv=15, return_estimator=True, scoring='accuracy')
best_idx = np.argmax(cv_results['test_score'])
best_model = cv_results['estimator'][best_idx]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vec.pkl')