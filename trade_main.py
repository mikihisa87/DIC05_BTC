
# coding: utf-8

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

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

#Model定義  *最適化済みパラメータ追記
clf_dt = tree.DecisionTreeClassifier(max_depth=1, criterion='gini')
clf_rf = ensemble.RandomForestClassifier(criterion='gini', max_depth=1, n_estimators=12)
clf_sgdc = SGDClassifier(alpha=0.0001,loss='log', max_iter=43, penalty='elasticnet',shuffle=False)
clf_svm = SVC(C=1, degree=2, gamma=0.001, kernel='poly')

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

#パラメータ最適化/Decision Tree
def dtbestparam(X, y):
    features = X
    targets = y
    #試行するパラメータを並べる
    params = {
        'max_depth' : list(range(1, 20)),
        'criterion' : ['gini', 'entropy'],
        }
    grid_search = GridSearchCV(clf_dt, #分類器を渡す
                                param_grid=params, #試行して欲しいパラメータを渡す
                                cv=10, # 10-Fold CVで汎化性能を調べる
                                )
    grid_search.fit(features, targets)
    return grid_search.best_score_, grid_search.best_params_
#結果
#best_score_=0.539726027397
#best_params_={'criterion': 'gini', 'max_depth': 1}

#パラメータ最適化/ensemble.RandomForestClassifier
def rfbestparam(X, y):
    features = X
    targets = y
    #試行するパラメータを並べる
    params = {
        'max_depth' : list(range(1, 20)),
        'criterion' : ['gini', 'entropy'],
        'n_estimators' : list(range(1, 20)),
        }
    grid_search = GridSearchCV(clf_rf, #分類器を渡す
                                param_grid=params, #試行して欲しいパラメータを渡す
                                cv=10, # 10-Fold CVで汎化性能を調べる
                                )
    grid_search.fit(features, targets)
    return grid_search.best_score_, grid_search.best_params_
#結果
#best_score_=0.627397260274
#best_params_={'criterion': 'gini', 'max_depth': 1, 'n_estimators': 12}

#パラメータ最適化/SGDClassifier
def sgdcbestparam(X, y):
    features = X
    targets = y
    #試行するパラメータを並べる
    params = {
        'loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive',
        'squared_epsilon_insensitive'],
        'penalty' : ['None', 'l2', 'l1', 'elasticnet'],
        'alpha' : list(np.arange(0.0001, 0.1, 10)),
        'max_iter' : list(range(5, 100)),
        'shuffle' : [False, True],
        }
    grid_search = GridSearchCV(clf_sgdc, #分類器を渡す
                                param_grid=params, #試行して欲しいパラメータを渡す
                                cv=10, # 10-Fold CVで汎化性能を調べる
                                )
    grid_search.fit(features, targets)
    return grid_search.best_score_, grid_search.best_params_
#結果
#best_score_=0.652054794521
#best_params_={'alpha': 0.0001, 'loss': 'log', 'max_iter': 43, 'penalty': 'elasticnet', 'shuffle': False}

#パラメータ最適化/SVM
def svmbestparam(X, y):
    features = X
    targets = y
    #試行するパラメータを並べる
    params = {
        'C' : [1, 10, 100, 1000],
        'kernel' : ['poly', 'rbf', 'sigmoid'],
        'degree' : [2, 3, 4],
        'gamma' : [0.001, 0.0001],
        }
    grid_search = GridSearchCV(clf_svm, #分類器を渡す
                                param_grid=params, #試行して欲しいパラメータを渡す
                                cv=10, # 10-Fold CVで汎化性能を調べる
                                )
    grid_search.fit(features, targets)
    return grid_search.best_score_, grid_search.best_params_
#結果
#best_score_=0.624657534247
#best_params_={'C': 1, 'degree': 2, 'gamma': 0.001, 'kernel': 'poly'}

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
accuracy(clf_svm, X, y, "SVM")
accuracy(clf_svm, X_norm, y, "SVM with Norm")
