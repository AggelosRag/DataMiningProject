# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 04:52:35 2020

@author: stavros bouras
"""
import sys
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict       
        
def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# preprocessing

df = pd.read_csv("onion-or-not.csv")

feature_cell=df[['text']]

yes_or_no=df[['label']] ##input for neural net

ps=PorterStemmer()

all_tokenized=[]
all_stemmed=[]

for index, row in df.iterrows():
    all_tokenized.append(nltk.word_tokenize(row['text']))
    
for i in all_tokenized:
    current_stemmed=[]
    for x in i:
        current_stemmed.append(ps.stem(x))
    all_stemmed.append(current_stemmed)

all_filtered=[]
         
for i in all_stemmed:
    stop_words = set(stopwords.words('english'))
    current_filtered = []
    for w in i:
        if w not in stop_words:
            current_filtered.append(w)
            
    all_filtered.append(current_filtered)

del all_stemmed,all_tokenized
 
uniqueWords=set()

for i in all_filtered:
    uniqueWords = uniqueWords.union(set(i))

all_dictionaries=[]

for i in all_filtered:
    
    my_dict= dict.fromkeys(uniqueWords, 0)
    
    for word in i:
        my_dict[word] += 1
        
    all_dictionaries.append(my_dict)


idf_list=[]

idf_list=computeIDF(all_dictionaries)

tf_idf_list=[]
for i in range(24000):
    tf_list= ((computeTF(all_dictionaries[i],all_filtered[i])))
    tf_idf_list.append(computeTFIDF(tf_list,idf_list))
  
del all_dictionaries,tf_list


final_array=[]

for d in tf_idf_list:
    temp=[]
    temp=list(d.values())
    final_array.append(temp) ##input for neural net
    
 # neural network
   
X=np.asarray(final_array,dtype=np.float16)
y=np.asarray(yes_or_no,dtype=np.int16)

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.25,random_state=4)

model = Sequential()
model.add(Dense(18000, input_dim=20456, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(20456, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m,precision_m, recall_m])

history = model.fit(X_train, y_train, epochs=4)

# # evaluate the model
loss,f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0,batch_size=32)

    
