
# coding: utf-8

# In[7]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import SGDClassifier


# In[2]:


os.chdir("./data")
dataset = pd.read_csv("Dataset.csv")
dataset.drop(columns={"Unnamed: 0", "date"}, inplace=True)
#dataset.set_index("date")


# In[11]:


#add label
price = dataset["price"]
pre_price = price.shift(-1)

labels = []
up = 1
down = 0
for i in range(len(dataset)):
    if price[i] <= pre_price[i]:
        labels.append(up)
    else:
        labels.append(down)
    
dataset["label"] = labels

#X,y定義
X = dataset.drop(columns=["label"])
y = dataset["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Model定義
clf_dt = tree.DecisionTreeClassifier()
clf_rf = ensemble.RandomForestClassifier()
clf_sgdc = SGDClassifier()

def testClf(clf, X_train, y_train, X_test, y_test):
    print(clf) 
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test), "\n")
    
testClf(clf_dt, X_train, y_train, X_test, y_test)
testClf(clf_rf, X_train, y_train, X_test, y_test)
testClf(clf_sgdc, X_train, y_train, X_test, y_test)

