
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

#os.chdir("./data")
dataset = pd.read_csv("Dataset.csv")
dataset.drop('Unnamed: 0', axis=1, inplace=True)
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

#パラメータ初期値
dt_param1 = 'gini'
dt_param2 = 1
rf_param1 = 'gini'
rf_param2 = 1
rf_param3 = 12
sgdc_param1 = 0.0001
sgdc_param2 = 'log'
sgdc_param3 = 43
sgdc_param4 = 'elasticnet'
sgdc_param5 = False
svm_param1 = 1
svm_param2 = 2
svm_param3 = 0.001
svm_param4 = 'poly'

#Model定義
clf_dt = tree.DecisionTreeClassifier(criterion=dt_param1, max_depth=dt_param2)
clf_rf = ensemble.RandomForestClassifier(criterion=rf_param1, max_depth=rf_param2, n_estimators=rf_param2)
clf_sgdc = SGDClassifier(alpha=sgdc_param1, loss=sgdc_param2, max_iter=sgdc_param3, penalty=sgdc_param4, shuffle=sgdc_param5)
clf_svm = SVC(C=svm_param1, degree=svm_param2, gamma=svm_param3, kernel=svm_param4)

#正規化なし
X = dataset.drop("label", axis=1)
y = dataset["label"]
X = X.set_index('date')

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
    print(grid_search.best_score_, grid_search.best_params_)
    return grid_search.best_params_


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
    print(grid_search.best_score_, grid_search.best_params_)
    return grid_search.best_params_

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
    print(grid_search.best_score_, grid_search.best_params_)
    return grid_search.best_params_


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
    print(grid_search.best_score_, grid_search.best_params_)
    return grid_search.best_params_

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
    return clf

#Xとx_norm(正規化されたX)を各モデルに入れ、ベストパラメータを表示
dtbestparam(X, y)
dtbestparam(X_norm, y)
#取得したベストパラメータに代入、モデルのパラメータを更新(X_normの学習値を使用)
dt_bestparam = dtbestparam(X_norm, y)
dt_param1 = dt_bestparam['criterion']
dt_param2 = dt_bestparam['max_depth']
#Xとx_norm(正規化されたX)を各モデルに入れ、ベストパラメータを表示
rfbestparam(X, y)
rfbestparam(X_norm, y)
#取得したベストパラメータに代入、モデルのパラメータを更新(X_normの学習値を使用)
rf_bestparam = rfbestparam(X_norm, y)
rf_param1 = rf_bestparam['criterion']
rf_param2 = rf_bestparam['max_depth']
rf_param3 = rf_bestparam['n_estimators']
#Xとx_norm(正規化されたX)を各モデルに入れ、ベストパラメータを表示
sgdcbestparam(X, y)
sgdcbestparam(X_norm, y)
#取得したベストパラメータに代入、モデルのパラメータを更新(X_normの学習値を使用)
sgdc_bestparam = sgdcbestparam(X_norm, y)
sgdc_param1 = sgdc_bestparam['alpha']
sgdc_param2 = sgdc_bestparam['loss']
sgdc_param3 = sgdc_bestparam['max_iter']
sgdc_param4 = sgdc_bestparam['penalty']
sgdc_param5 = sgdc_bestparam['shuffle']
#Xとx_norm(正規化されたX)を各モデルに入れ、ベストパラメータを表示
svmbestparam(X, y)
svmbestparam(X_norm, y)
#取得したベストパラメータに代入、モデルのパラメータを更新(X_normの学習値を使用)
svm_bestparam = svmcbestparam(X_norm, y)
svm_param1 = svm_bestparam['C']
svm_param2 = svm_bestparam['degree']
svm_param3 = svm_bestparam['gamma']
svm_param4 = svm_bestparam['kernel']

#Model再定義  
clf_dt = tree.DecisionTreeClassifier(criterion=dt_param1, max_depth=dt_param2)
clf_rf = ensemble.RandomForestClassifier(criterion=rf_param1, max_depth=rf_param2, n_estimators=rf_param2)
clf_sgdc = SGDClassifier(alpha=sgdc_param1, loss=sgdc_param2, max_iter=sgdc_param3, penalty=sgdc_param4, shuffle=sgdc_param5)
clf_svm = SVC(C=svm_param1, degree=svm_param2, gamma=svm_param3, kernel=svm_param4)

#精度表示
accuracy(clf_dt, X, y, "DecisionTree")
accuracy(clf_dt, X_norm, y, "DecisionTree with Norm")
accuracy(clf_rf, X, y, "RamdomForest")
accuracy(clf_rf, X_norm, y, "RamdomForest with Norm")
accuracy(clf_sgdc, X, y, "SGDC")
accuracy(clf_sgdc, X_norm, y, "SGDC with Norm")
accuracy(clf_svm, X, y, "SVM")
accuracy(clf_svm, X_norm, y, "SVM with Norm")
