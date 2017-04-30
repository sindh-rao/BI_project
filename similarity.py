# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.neighbors import DistanceMetric
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn import tree


df = pd.read_csv("../input/train.csv",nrows=200)
#df = pd.read_csv("../input/train.csv",nrows = 20000)
df = df.dropna()

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

words = re.compile(r"\w+",re.I)
stopword = stopwords.words('english')
#stemmer = PorterStemmer()

def tokenize(df):
    q1 = []
    q2 = []
    for q in df.question1.tolist():
        q1.append([(i.lower()) for i in words.findall(q) if i not in stopword])
    for q in df.question2.tolist():
        q2.append([(i.lower()) for i in words.findall(q) if i not in stopword])
    df["q1_tokens"] = q1
    df["q2_tokens"] = q2
    return df

def train_dictionary(df):
    q_tokens = df.q1_tokens.tolist() + df.q2_tokens.tolist()
    dictionary = corpora.Dictionary(q_tokens)
    return dictionary
    
df = tokenize(df)
dictionary = train_dictionary(df)

def get_vectors(df, dictionary):
    
    question1_vec = [dictionary.doc2bow(text) for text in df.q1_tokens.tolist()]
    question2_vec = [dictionary.doc2bow(text) for text in df.q2_tokens.tolist()]
    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
    return question1_csc.transpose(),question2_csc.transpose()
q1_csc, q2_csc = get_vectors(df, dictionary)

minkowski_dis = DistanceMetric.get_metric('minkowski')

def get_similarity_values(q1_csc, q2_csc):
    cosine_sim = []
    manhattan_dis = []
    eucledian_dis = []
    jaccard_dis = []
    minkowsk_dis = []
    
    for i,j in zip(q1_csc, q2_csc):
        sim = cs(i,j)
        cosine_sim.append(sim[0][0])
        sim = md(i,j)
        manhattan_dis.append(sim[0][0])
        sim = ed(i,j)
        eucledian_dis.append(sim[0][0])
        x = i.toarray()
        y = j.toarray()
        try:
            sim = jsc(x,y)
            jaccard_dis.append(sim)
        except:
            jaccard_dis.append(0)
            
        sim = minkowski_dis.pairwise(x,y)
        minkowsk_dis.append(sim[0][0])
    
    return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis    

cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis = get_similarity_values(q1_csc, q2_csc)

df["cosine"] = cosine_sim
df["manhattan"] = manhattan_dis
df["eucledian"] = jaccard_dis
df["minkowsk"] = minkowsk_dis
df["jaccard"] = jaccard_dis

print(df.head())
reg = linear_model.LogisticRegression()
svc = SVC()
dt = tree.DecisionTreeClassifier()
df_train, df_test = train_test_split(df, test_size = 0.3)
ytrain = df_train["is_duplicate"]
ytest = df_test["is_duplicate"]
#print(df_train.head())
xtrain = df_train.ix[:,'cosine':]
xtest = df_test.ix[:,'cosine':]
#print(xtrain.head())
lr_model=reg.fit(np.array(xtrain), np.array(ytrain))
svc_model = svc.fit(np.array(xtrain),np.array(ytrain))
dt_model = dt.fit(np.array(xtrain),np.array(ytrain))

Ypred_lr=lr_model.predict(np.array(xtest))
Ypred_svc = svc_model.predict(np.array(xtest))
Ypred_dt = dt_model.predict(np.array(xtest))

accuracy_lr=accuracy_score(ytest, Ypred_lr)
accuracy_svc = accuracy_score(ytest,Ypred_svc)
accuracy_dt = accuracy_score(ytest,Ypred_dt)

print("Accuracy of logistic model",accuracy_lr)
print("Accuracy of SVM model", accuracy_svc)
print("Accuracy of decision tree classifier model", accuracy_dt)

print("log loss values for logistic model", log_loss(ytest,lr_model.predict_proba(np.array(xtest))))
print("log loss values for decision model", log_loss(ytest,dt_model.predict_proba(np.array(xtest))))                                                     