#!/usr/bin/env python
# coding: utf-8

# In[1]: Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot
import pylab
import random 
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing 

# In[2]: Erotima A

data = pd.read_csv("winequality-red.csv")
x = data.drop('quality', axis='columns')
y = data.quality

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y,test_size=0.25)

model = SVC(C= 100, gamma= 'auto', kernel= 'rbf')
model.fit(x_tr, y_tr)
y_pred=model.predict(x_ts)
print(classification_report(y_ts, y_pred))


# In[3]: Erotima B1

data = pd.read_csv("winequality-red.csv")
x = data.drop(['quality', 'pH'], axis='columns')
y = data.quality

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y,test_size=0.25)

model = SVC(C= 100, gamma= 'auto', kernel= 'rbf');
model.fit(x_tr, y_tr)
y_pred=model.predict(x_ts)
print(classification_report(y_ts, y_pred))

matplotlib.pyplot.scatter(data.quality,data.pH)
matplotlib.pyplot.show()
print(data['quality'].value_counts())

# In[4]: Erotima B2

data = pd.read_csv("winequality-red.csv")
x = data.drop('quality', axis='columns')
y = data.quality

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y,test_size=0.25)

sum = 0
for k in x_tr.pH:
    sum = sum + k
mean = sum/x_tr.pH.count()

for r in range (0,396):
    i  = random.randint(0,1200)
    while (i not in x_tr.pH.index or x_tr.pH[i] == mean): 
        i  = random.randint(0,1200)
    x_tr.pH[i] = mean

model = SVC(C= 100, gamma= 'auto', kernel= 'rbf')
model.fit(x_tr, y_tr)
y_pred=model.predict(x_ts)
print(classification_report(y_ts, y_pred))

# In[5]: Erotima B3

data = pd.read_csv("winequality-red.csv")
x = data.drop('quality', axis='columns')
y = data.quality

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y,test_size=0.25, random_state=2)
x_tr.loc[x_tr.sample(frac=0.33,random_state=4).index,'pH'] = np.nan

label_encoder = preprocessing.LabelEncoder()
samples = x_tr.sample(frac=0.33,random_state=4).index

y_tr_with_pH = label_encoder.fit_transform((data[~data.index.isin(samples)])['pH'])
x_tr_with_pH = (data[~data.index.isin(samples)]).drop('pH', axis=1)
y_pH_pred = (data[data.index.isin(samples)])['pH']
x_tr_without_pH = (data[data.index.isin(samples)]).drop('pH', axis=1)

model = LogisticRegression()
model.fit(x_tr_with_pH, y_tr_with_pH)
y_pH_pred = model.predict(x_tr_without_pH)
y_pH_pred = label_encoder.inverse_transform(y_pH_pred)
cnt = 0;

for index in x_tr_without_pH.index:
    (x_tr.loc[index])['pH']=y_pH_pred[cnt]
    cnt=cnt+1
    
model = SVC(C= 100, gamma= 'auto', kernel= 'rbf')
model.fit(x_tr, y_tr)
y_pred=model.predict(x_ts)
print(classification_report(y_ts, y_pred))


# In[6]: Erotima B4

data=pd.read_csv('winequality-red.csv')
feature_df = data.drop("pH", axis=1)

data.loc[data.sample(frac=0.33,random_state=4).index,'pH'] = np.nan
    
wcss=[]    
for i in range(1,16):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(feature_df)
    wcss.append(kmeans.inertia_)
    
matplotlib.pyplot.plot(range(1,16),wcss)
matplotlib.pyplot.title('Test')
matplotlib.pyplot.xlabel('Number Of Clusters')
matplotlib.pyplot.ylabel('WCSS')
matplotlib.pyplot.show()


kmeans=KMeans(n_clusters=4, init='k-means++',random_state=0)
y=kmeans.fit_predict(feature_df)

feature_df['Clusters']=y
k1=feature_df[y==0]
k1.insert(8,'pH',data['pH'])
k2=feature_df[y==1]
k2.insert(8,'pH',data['pH'])
k3=feature_df[y==2]
k3.insert(8,'pH',data['pH'])
k4=feature_df[y==3]
k4.insert(8,'pH',data['pH'])

k1.fillna(k1.mean(), inplace=True)
k2.fillna(k2.mean(), inplace=True)
k3.fillna(k3.mean(), inplace=True)
k4.fillna(k4.mean(), inplace=True)

new_celldf1=pd.concat([k1,k2,k3,k4],ignore_index=False)
del new_celldf1['Clusters']

new_celldf2=new_celldf1[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]

X1=np.asarray(new_celldf2)
y1=np.asarray(new_celldf1['quality'])
X_train, X_test, y_train, y_test= train_test_split(X1,y1,test_size= 0.25,random_state=4)

classifier = SVC(C= 100, gamma= 'auto', kernel= 'rbf')
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)

print(classification_report(y_test,y_predict))
