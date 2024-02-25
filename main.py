# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_csv_path = "/kaggle/input/dataset/Train.csv"
test_csv_path = "/kaggle/input/dataset/Test.csv"

train_data = pd.read_csv(train_csv_path, nrows=100000)
test_data = pd.read_csv(test_csv_path)

pip install nltk

corpus = []
ps = PorterStemmer()

for i in range(len(train_data)):
    review = re.sub('[^a-zA-Z]', ' ', train_data['review_text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
x_train = tfidf_vectorizer.fit_transform(corpus).toarray()

y_train=train_data['rating']

from sklearn.linear_model import LogisticRegression


classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_train)

from sklearn.metrics import accuracy_score
accuracy_score(y_train,y_pred)

import lightgbm as lgb

# LightGBM model
lgb_classifier = lgb.LGBMClassifier(random_state=0)
lgb_classifier.fit(x_train, y_train)

# Predictions on the test set using LightGBM
y_pred_lgb = lgb_classifier.predict(x_train)

# Calculate and print accuracy for LightGBM
accuracy_lgb = accuracy_score(y_train, y_pred_lgb)
print(f"LightGBM Accuracy: {accuracy_lgb}")

import pickle
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(corpus, f)

with open('preprocessed_data.pkl', 'rb') as f:
    corpus = pickle.load(f)
test_corpus = []
# ps = PorterStemmer()

for i in range(len(test_data)):
    review = re.sub('[^a-zA-Z]', ' ', test_data['review_text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = " ".join(review)
    test_corpus.append(review)

with open('preprocessed_data1.pkl', 'wb') as f:
     pickle.dump(test_corpus, f)


with open('preprocessed_data1.pkl', 'rb') as f:
    test_corpus = pickle.load(f)

x_test = tfidf_vectorizer.fit_transform(test_corpus).toarray()


y_pred_lgb_test = lgb_classifier.predict(x_test)


review_ids=test_data['review_id'].tolist()


submission_df=pd.DataFrame({'review_id':review_ids,'rating':y_pred_lgb_test})



submission_df.to_csv('submission.csv',index=False)
