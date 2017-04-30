import pandas as pd 
import string
import sys
import csv
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import collections
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import numpy as np
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import nltk
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
from nltk.stem import SnowballStemmer
import re
from sklearn.model_selection import train_test_split

data = pd.read_csv('questions.csv')
##data = data.ix[:1000, :]
data.dropna()

stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation 


##preprocessing
def preprocess(text, remove_stop_words=True, stem_words=False):

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

##Generate labeled sentences
def generate_label_list(listofwords):
    lslist = []
    for line in listofwords:
        lslist.append(LabeledSentence(words = line[0], tags=line[1]))
    return lslist

data['question1'] = data['question1'].apply(lambda x: unicode(str(x),"utf-8"))
data['question2'] = data['question2'].apply(lambda x: unicode(str(x),"utf-8"))


data['question1_preprocess'] = data.question1.apply(preprocess)
data['question2_preprocess'] = data.question2.apply(preprocess)
questions = list(data['question1_preprocess']) + list(data['question2_preprocess'])

##y_train = train['is_duplicate']
##y_test = test['is_duplicate']
##print(y_train)

sentences = []
for index, row in data.iterrows():
  wordsq1 = [w for w in row['question1_preprocess'].strip().split()]
  wordsq2 = [w for w in row['question2_preprocess'].strip().split()]
  sentences.append([wordsq1,str(row['qid1'])])
  sentences.append([wordsq2,str(row['qid2'])])


labeled_sent=generate_label_list(sentences)


##Doc2Vec model
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
model.build_vocab(labeled_sent) 
for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(labeled_sent)
        model.train(labeled_sent)
 
docvecs = model.docvecs

##Similarity scores
sim_scores = []
for index, row in data.iterrows(): 
   if not row['question1_preprocess'].strip().split():
     score = 0
   elif not row['question2_preprocess'].strip().split():
     score = 0
   else:
     score = model.n_similarity(row['question1_preprocess'].strip().split(),row['question2_preprocess'].strip().split())  
   sim_scores.append(score)
data['sim'] = sim_scores     

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

##Regression
clf1=LogisticRegression()
lr_model=clf1.fit(np.array(x_train).reshape(-1,1), np.array(y_train))
Ypred=lr_model.predict(np.array(x_test).reshape(-1,1))
Ypredprob=lr_model.predict_proba(np.array(x_test).reshape(-1,1))

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
cvscores = cross_val_score(clf1, data.sim.values.reshape(-1,1),data.is_duplicate, cv=10)
cvscores = [ convert(x) for x in cvscores ]

errorFile = open("errors.csv", 'wb')
wr = csv.writer(errorFile, delimiter=";")
for error in cvscores:
  wr.writerow(["doc2vec", error])



##Reference: https://www.kaggle.com/quora/question-pairs-dataset/kernels




