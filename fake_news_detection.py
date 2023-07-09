import pandas as pd  #for csv file  
import numpy as np  #for conversion to array    
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB   # using nulti naive bayes model to predict


data = pd.read_csv("fake_or_real_news.csv")

x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()

x = cv.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=65)

model = MultinomialNB()

model.fit(xtrain,ytrain)

# print(model.score(xtest,ytest)) to test the accuracy of model

import streamlit as st 
# using the streamlit library in Python to build an 
# end-to-end application for the machine learning model to detect fake news in real-time

st.title("Fake News Detection System")

def fakenewsdetection():
    user = st.text_area("Enter any NEWS Headline: ")
    if len(user) < 1:
        st.write(" ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)

fakenewsdetection()
