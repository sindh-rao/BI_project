import string
import sys
import re
import csv
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer
from sklearn import svm
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import numpy as np
import random
##random.seed(0)
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np
from nltk import word_tokenize
from scipy.spatial.distance import cosine
import pandas as pd
##from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense
##from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
nlp = spacy.load('en')

stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation 

def preprocess(text, remove_stop_words=False, stem_words=False):

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_list]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

##read data
data = pd.read_csv('questions.csv')
##data = data.ix[:1000, :]
data.dropna() 
 
data['question1'] = data['question1'].apply(lambda x: unicode(str(x),"utf-8"))
data['question2'] = data['question2'].apply(lambda x: unicode(str(x),"utf-8"))

data['question1_preprocess'] = data.question1.apply(preprocess)
data['question2_preprocess'] = data.question2.apply(preprocess)
questions = list(data['question1_preprocess']) + list(data['question2_preprocess'])


##averaged tf-idf weighed word vectors for question1
q1vecs = []
for q in list(data['question1_preprocess']):
    doc = nlp(q)
    mean_words_vec = np.zeros(300) 
    for word in doc:
        vec = word.vector   ##spacys wordtovec
        mean_words_vec += vec 
    norm = np.linalg.norm(mean_words_vec)
    if norm!=0:
      mean_words_vec /= norm
    q1vecs.append(mean_words_vec)
data['q1vecs'] = q1vecs

##averaged tf-idf weighed word vectors for question2
q2vecs = []
for q in list(data['question2_preprocess']):
    doc = nlp(q)
    mean_words_vec = np.zeros(300) 
    for word in doc:
        vec = word.vector   ##spacys wordtovec
        mean_words_vec += vec 
    norm = np.linalg.norm(mean_words_vec)
    if norm!=0:
      mean_words_vec /= norm
    q2vecs.append(mean_words_vec)
data['q2vecs'] = q2vecs
print len(q1vecs)
print len(q2vecs)

##cosine_similarity_scores
scores=[]
for i in range(len(q1vecs)):
    ##cosine_similarity = np.dot(q1vecs[i], q2vecs[i])/(np.linalg.norm(q1vecs[i])* np.linalg.norm(q2vecs[i]))
    cosine_similarity = np.dot(q1vecs[i], q2vecs[i])/np.sqrt(np.dot(q1vecs[i], q2vecs[i]))*np.sqrt(np.dot(q1vecs[i], q2vecs[i]))
    scores.append(cosine_similarity)
data['sim'] = scores 



##split test_train
test, train = train_test_split(data, test_size = 0.2)
train.dropna()
test.dropna()

##generate input and output train and test vectors 
y_train = []
y_test = []

x_train = []
for index, row in train.iterrows(): 
   x_train.append(row['sim'])
   y_train.append(row['is_duplicate'])

x_test = []
for index, row in test.iterrows(): 
   x_test.append(row['sim'])
   y_test.append(row['is_duplicate'])

imp = Imputer(missing_values='NaN', strategy='median', axis=1) 

x_train = imp.fit_transform(x_train)
x_test =imp.fit_transform(x_test)

##Logistic Regression model for classification
clf1=LogisticRegression()
lr_model=clf1.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))
Ypred=lr_model.predict(np.array(x_test).reshape(-1,1))
Ypredprob=lr_model.predict_proba(np.array(x_test).reshape(-1,1))

##Model metrics
accuracy=accuracy_score(y_test, Ypred)
print("accuracy: ")
print(accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, Ypred))
    
print("Classification Report:")
print(classification_report(y_test, Ypred))

logloss = log_loss(y_test, Ypredprob)
print("logloss:")
print(logloss)


def convert(score):
  return (1-score)*100
simvalues = data['sim']
simvalues=imp.fit_transform(simvalues)
cvscores = cross_val_score(clf1, simvalues.reshape(-1,1),data.is_duplicate.values, cv=10)
cvscores = [ convert(x) for x in cvscores ]


fd = open('errors.csv','a')
writer = csv.writer(fd, delimiter=";")
for error in cvscores:
  writer.writerow(["word2vectfidf",error])
fd.close()


##Reference: https://www.kaggle.com/quora/question-pairs-dataset/kernels
##Reference: http://www.erogol.com/duplicate-question-detection-deep-learning/

