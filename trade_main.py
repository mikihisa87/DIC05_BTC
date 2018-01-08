
# coding: utf-8

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import SGDClassifier

os.chdir("./data")
dataset = pd.read_csv("Dataset.csv")
dataset.drop(columns={"Unnamed: 0", "date"}, inplace=True)
#dataset.set_index("date")

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

#Model定義
clf_dt = tree.DecisionTreeClassifier()
clf_rf = ensemble.RandomForestClassifier()
clf_sgdc = SGDClassifier()

#正規化なし
X = dataset.drop(columns=["label"])
y = dataset["label"]

#正規化あり
X_array = np.array(X)
def  zscore(X, axis=None):
    xmean = X.mean(axis=axis, keepdims=True)
    xstd = np.std(X, axis=axis, keepdims=True)
    zscore = (X-xmean)/xstd
    return zscore

X_norm = zscore(X_array)

#精度確認
def accuracy(clf, X, y, model): 
    cnt = 0
    score_all = []
    while cnt < 100:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#random_state=0
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        score_all.append(score)
        cnt +=1

    score_all = np.array(score_all)
    mscore = (score_all.mean())*100
    print("{} Mean_Score :".format(model), round(mscore, 2), "%")
    return

accuracy(clf_dt, X, y, "DecisionTree")
accuracy(clf_dt, X_norm, y, "DecisionTree with Norm")
accuracy(clf_rf, X, y, "RamdomForest")
accuracy(clf_rf, X_norm, y, "RamdomForest with Norm")
accuracy(clf_sgdc, X, y, "SGDC")
accuracy(clf_sgdc, X_norm, y, "SGDC with Norm")
