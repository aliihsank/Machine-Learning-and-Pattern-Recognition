# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:00:03 2021

@author: ali_k
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def probability(prior, likelihood1, likelihood2, likelihood3, likelihood4, likelihood5, likelihood6, likelihood7, likelihood8):
    return prior * likelihood1 * likelihood2 * likelihood3 * likelihood4 * likelihood5 * likelihood6 * likelihood7 * likelihood8


def likelihood_dist(column, h):
	return lambda x: len([i for i in column if (x+h)>i>(x-h)]) / (2 * len(column) * h)

headers = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscara_weight', 'shell_weight', 'class']

dataset = pd.read_csv('abalone_dataset.txt', names=headers, delimiter='	')

X = dataset.iloc[:, :8]
y = dataset.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, random_state=26, shuffle=True)

Xy1 = X_train[y_train == 1]
Xy2 = X_train[y_train == 2]
Xy3 = X_train[y_train == 3]

prior1 = len(Xy1)/len(X_train)
prior2 = len(Xy2)/len(X_train)
prior3 = len(Xy3)/len(X_train)

X1y1 = likelihood_dist(Xy1.iloc[:, 1], 0.3)
X2y1 = likelihood_dist(Xy1.iloc[:, 2], 0.3)
X3y1 = likelihood_dist(Xy1.iloc[:, 3], 0.3)
X4y1 = likelihood_dist(Xy1.iloc[:, 4], 0.3)
X5y1 = likelihood_dist(Xy1.iloc[:, 5], 0.3)
X6y1 = likelihood_dist(Xy1.iloc[:, 6], 0.3)
X7y1 = likelihood_dist(Xy1.iloc[:, 7], 0.3)

X1y2 = likelihood_dist(Xy2.iloc[:, 1], 0.3)
X2y2 = likelihood_dist(Xy2.iloc[:, 2], 0.3)
X3y2 = likelihood_dist(Xy2.iloc[:, 3], 0.3)
X4y2 = likelihood_dist(Xy2.iloc[:, 4], 0.3)
X5y2 = likelihood_dist(Xy2.iloc[:, 5], 0.3)
X6y2 = likelihood_dist(Xy2.iloc[:, 6], 0.3)
X7y2 = likelihood_dist(Xy2.iloc[:, 7], 0.3)

X1y3 = likelihood_dist(Xy3.iloc[:, 1], 0.3)
X2y3 = likelihood_dist(Xy3.iloc[:, 2], 0.3)
X3y3 = likelihood_dist(Xy3.iloc[:, 3], 0.3)
X4y3 = likelihood_dist(Xy3.iloc[:, 4], 0.3)
X5y3 = likelihood_dist(Xy3.iloc[:, 5], 0.3)
X6y3 = likelihood_dist(Xy3.iloc[:, 6], 0.3)
X7y3 = likelihood_dist(Xy3.iloc[:, 7], 0.3)

y_pred = []
for line in X_train.values:
    prob1 = probability(prior1, len(Xy1[Xy1.iloc[:, 0] == line[0]]) / len(Xy1), X1y1(line[1]), X2y1(line[2]), X3y1(line[3]), X4y1(line[4]), X5y1(line[5]), X6y1(line[6]), X7y1(line[7]))
    prob2 = probability(prior2, len(Xy2[Xy2.iloc[:, 0] == line[0]]) / len(Xy2), X1y2(line[1]), X2y2(line[2]), X3y2(line[3]), X4y2(line[4]), X5y2(line[5]), X6y2(line[6]), X7y2(line[7]))
    prob3 = probability(prior3, len(Xy3[Xy3.iloc[:, 0] == line[0]]) / len(Xy3), X1y3(line[1]), X2y3(line[2]), X3y3(line[3]), X4y3(line[4]), X5y3(line[5]), X6y3(line[6]), X7y3(line[7]))
    
    predicted = 1
    if prob1 > prob2 and prob1 > prob3:
        predicted = 1
    elif prob2 > prob1 and prob2 > prob3:
        predicted = 2
    elif prob3 > prob1 and prob3 > prob2:
        predicted = 3
    
    y_pred.append(predicted)
    

X_train['prediction'] = y_pred

acc = accuracy_score(y_train, y_pred)
print('Train ACC: ', acc)
cm = confusion_matrix(y_train, y_pred)
print('Train CM: ', cm)
numOfMissclassifiedTrain = len(X_train) - np.trace(cm)
print('# of Missclasification(Train): ', numOfMissclassifiedTrain)

y_pred = []
for line in X_test.values:
    prob1 = probability(prior1, len(Xy1[Xy1.iloc[:, 0] == line[0]]) / len(Xy1), X1y1(line[1]), X2y1(line[2]), X3y1(line[3]), X4y1(line[4]), X5y1(line[5]), X6y1(line[6]), X7y1(line[7]))
    prob2 = probability(prior2, len(Xy2[Xy2.iloc[:, 0] == line[0]]) / len(Xy2), X1y2(line[1]), X2y2(line[2]), X3y2(line[3]), X4y2(line[4]), X5y2(line[5]), X6y2(line[6]), X7y2(line[7]))
    prob3 = probability(prior3, len(Xy3[Xy3.iloc[:, 0] == line[0]]) / len(Xy3), X1y3(line[1]), X2y3(line[2]), X3y3(line[3]), X4y3(line[4]), X5y3(line[5]), X6y3(line[6]), X7y3(line[7]))
    
    predicted = 1
    if prob1 > prob2 and prob1 > prob3:
        predicted = 1
    elif prob2 > prob1 and prob2 > prob3:
        predicted = 2
    elif prob3 > prob1 and prob3 > prob2:
        predicted = 3
    
    y_pred.append(predicted)
    

X_test['prediction'] = y_pred

acc = accuracy_score(y_test, y_pred)
print('Test ACC: ', acc)
cm = confusion_matrix(y_test, y_pred)
print('Test CM: ', cm)
numOfMissclassifiedTest = len(X_test) - np.trace(cm)
print('# of Missclasification(Test): ', numOfMissclassifiedTest)
    
    