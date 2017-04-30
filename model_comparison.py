import csv
from decimal import *
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import numpy as np
data = pd.read_csv('errors.csv')

##Check normal distribution
print(stats.normaltest(data[data['Model'] == 'doc2vec'].error_value))
print(stats.normaltest(data[data['Model'] == 'word2vectfidf'].error_value))
print(stats.normaltest(data[data['Model'] == 'word2vec'].error_value))

##Bartlett test to test homoscedascity
array1=np.array(data[data['Model'] == 'doc2vec'].error_value)
array2=np.array(data[data['Model'] == 'word2vectfidf'].error_value)
array3=np.array(data[data['Model'] == 'word2vectfidf'].error_value)
print(stats.bartlett(array1, array2, array3))

##One-way anova test
f, p = stats.f_oneway(data[data['Model'] == 'doc2vec'].error_value,
                      data[data['Model'] == 'word2vectfidf'].error_value,
                      data[data['Model'] == 'word2vec'].error_value)
 
print ('One-way ANOVA')
print ('=============')
 
print ('F value:', f)
print ('P value:', p, '\n')

##two sample t-test for pair-wise comparison.
print("Doc2Vec-word2vectfidf")
print(stats.ttest_ind(data[data['Model'] == 'doc2vec'].error_value, data[data['Model'] == 'word2vectfidf'].error_value))
print("word2vectfidf-word2vec")
print(stats.ttest_ind(data[data['Model'] == 'word2vectfidf'].error_value,data[data['Model'] == 'word2vec'].error_value))
print("word2vec-doc2vec")
print(stats.ttest_ind(data[data['Model'] == 'word2vec'].error_value, data[data['Model'] == 'doc2vec'].error_value))

print("Doc2vec mean")
print(data[data['Model'] == 'doc2vec'].error_value.mean())
print("word2vectfidf")
print(data[data['Model'] == 'word2vectfidf'].error_value.mean())
print("word2vec")
print(data[data['Model'] == 'word2vec'].error_value.mean())
