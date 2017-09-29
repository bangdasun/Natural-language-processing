# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:38:50 2017

@author: Bangda

To run this file, type in terminal or cmd
    python classify train.txt test.txt
    
This file reads raw tweets data and apply unigram/bigram/trigram to extract data
classify whether the author of tweets is Republican or Democrat
Algorithms include: Multinomial Naive Bayes; Logistics Regression; Linear SVM (other kernel will take longer time to train)

To be continued... 
There is still space avaiable for improvment!
"""

import sys

import numpy as np
import re

### load data
with open(sys.argv[2], encoding = 'latin-1') as dev_con:
    dev = np.loadtxt(dev_con, dtype = str, delimiter = '\\s', comments = None)

with open(sys.argv[1], encoding = 'latin-1') as train_con:
    train = np.loadtxt(train_con, dtype = str, delimiter = '\\s', comments = None)

### formatting data
train_data = []
dev_data   = []

for i in range(len(train)):
     train_data.append(train[i].split('\\t'))
     
for i in range(len(dev)):
    dev_data.append(dev[i].split('\\t'))
    
train_data = np.array(train_data)
dev_data   = np.array(dev_data)

X_train, y_train = train_data[:, 0], train_data[:, 1]
X_dev, y_dev = dev_data[:, 0], dev_data[:, 1]

for i in range(len(y_train)):
    if y_train[i][0] == 'd':
        y_train[i] = 1
    else:
        y_train[i] = 0
        
for i in range(len(y_dev)):
    if y_dev[i][0] == 'd':
        y_dev[i] = 1
    else:
        y_dev[i] = 0
        
y_train = np.array(y_train, dtype = float)
y_dev   = np.array(y_dev, dtype = float)


### inspect data
X_train.shape, y_train.shape, X_dev.shape, y_dev.shape


### Extract stylized features
##  1. lengthening word (sooooo coool), count
lengthening_word_count_train = [len(re.findall(r'([A-Za-z])\1{3,}', x)) for x in X_train]
lengthening_word_count_dev   = [len(re.findall(r'([A-Za-z])\1{3,}', x)) for x in X_dev]

##  2. all upper case word (AWESOME), count
all_upper_count_train = [len(re.findall(r'[A-Z]{3,}', x)) for x in X_train]
all_upper_count_dev   = [len(re.findall(r'[A-Z]{3,}', x)) for x in X_dev]

##  3. multi punctations (!!! or ???), count
multi_punct_count_train = [len(re.findall(r'(!\s*|\?\s*){2,}', x)) for x in X_train]
multi_punct_count_dev   = [len(re.findall(r'(!\s*|\?\s*){2,}', x)) for x in X_dev]

##  4. hashtag (#update), count
hashtag_count_train = [len(re.findall(r'#', x)) for x in X_train]
hashtag_count_dev   = [len(re.findall(r'#', x)) for x in X_dev]


### Count emojis
#   emojis are represented by their meanings with underscore connection, e.g. thumb_up
emoji_count_train = [len(re.findall(r'[0-9a-zA-Z]+(?:_[a-zA-Z]+)+', x)) - 
                     len(re.findall(r'@[0-9a-zA-Z]+(?:_[a-zA-Z]+)+', x)) 
                     for x in X_train]
emoji_count_dev   = [len(re.findall(r'[0-9a-zA-Z]+(?:_[a-zA-Z]+)+', x)) - 
                     len(re.findall(r'@[0-9a-zA-Z]+(?:_[a-zA-Z]+)+', x)) 
                     for x in X_dev]


### Extract unigram / bigram features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS.union(['http', 'just', 'like', 'day', 'time'])


##  1. unigram
unig_vectorizer = CountVectorizer(stop_words = stop_words, ngram_range = (1, 1))
X_train_unig = unig_vectorizer.fit_transform(X_train)
X_dev_unig = unig_vectorizer.transform(X_dev)

##  2. bigram
big_vectorizer = CountVectorizer(stop_words = stop_words, ngram_range = (2, 2))
X_train_big = big_vectorizer.fit_transform(X_train)
X_dev_big = big_vectorizer.transform(X_dev)

##  3. trigram
trig_vectorizer = CountVectorizer(stop_words = stop_words, ngram_range = (3, 3))
X_train_trig = trig_vectorizer.fit_transform(X_train)
X_dev_trig = trig_vectorizer.transform(X_dev)


### Append extra features to the sparse matrix produced by n-gram model
from scipy.sparse import hstack, csr_matrix

def append_extra_feature(X_train, lst):
    stack_lst = [X_train]
    for lst_i in lst:
        lst_i = csr_matrix(lst_i)
        stack_lst.append(lst_i.T)
    
    return hstack(stack_lst)

##  train
append_lst_train = [lengthening_word_count_train, all_upper_count_train, 
                    multi_punct_count_train, hashtag_count_train, emoji_count_train]

X_train_unig_aug = append_extra_feature(X_train_unig, append_lst_train)
X_train_big_aug  = append_extra_feature(X_train_big, append_lst_train)
X_train_trig_aug = append_extra_feature(X_train_trig, append_lst_train)

##  dev
append_lst_dev = [lengthening_word_count_dev, all_upper_count_dev, 
                  multi_punct_count_dev, hashtag_count_dev, emoji_count_dev]

X_dev_unig_aug = append_extra_feature(X_dev_unig, append_lst_dev)
X_dev_big_aug  = append_extra_feature(X_dev_big, append_lst_dev)
X_dev_trig_aug = append_extra_feature(X_dev_trig, append_lst_dev)


### Modeling
from sklearn.metrics import accuracy_score

##  1. Multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB

# a) unigram - naive bayes
nb_unig = MultinomialNB(alpha = .11)
nb_unig.fit(X_train_unig_aug, y_train)
y_pred_dev = nb_unig.predict(X_dev_unig_aug)
print("Accuracy on test set of unigram naive bayes: {}".format(accuracy_score(y_pred_dev, y_dev)))

# b) bigram - naive bayes
nb_big = MultinomialNB(alpha = .11)
nb_big.fit(X_train_big_aug, y_train)
y_pred_dev = nb_big.predict(X_dev_big_aug)
print("Accuracy on test set of bigram naive bayes: {}".format(accuracy_score(y_pred_dev, y_dev)))

# c) trigram - naive bayes
nb_trig = MultinomialNB(alpha = .11)
nb_trig.fit(X_train_trig_aug, y_train)
y_pred_dev = nb_trig.predict(X_dev_trig_aug)
print("Accuracy on test set of trigram naive bayes: {}".format(accuracy_score(y_pred_dev, y_dev)))


##  2. Logistic regression
from sklearn.linear_model import LogisticRegression

# a) unigram - logistic regression
lr_unig = LogisticRegression(C = 1.5, fit_intercept = False, penalty = 'l2')
lr_unig.fit(X_train_unig_aug, y_train)
y_pred_dev = lr_unig.predict(X_dev_unig_aug)
print("Accuracy on test set of unigram logistic regression: {}".format(accuracy_score(y_pred_dev, y_dev)))

# b) bigram - logistic regression
lr_big = LogisticRegression(C = 1.5, fit_intercept = False, penalty = 'l2')
lr_big.fit(X_train_big_aug, y_train)
y_pred_dev = lr_big.predict(X_dev_big_aug)
print("Accuracy on test set of bigram logistic regression: {}".format(accuracy_score(y_pred_dev, y_dev)))

# c) trigram - logistic regression
lr_trig = LogisticRegression(C = 1.5, fit_intercept = False, penalty = 'l2')
lr_trig.fit(X_train_trig_aug, y_train)
y_pred_dev = lr_trig.predict(X_dev_trig_aug)
print("Accuracy on test set of trigram logistic regression: {}".format(accuracy_score(y_pred_dev, y_dev)))


## 3. SVC (linear kernel)
from sklearn.svm import LinearSVC

# a) unigram - linear svc
lin_svc_unig = LinearSVC(C = .13)
lin_svc_unig.fit(X_train_unig_aug, y_train)
y_pred_dev = lin_svc_unig.predict(X_dev_unig_aug)
print("Accuracy on test set of unigram linear svc: {}".format(accuracy_score(y_pred_dev, y_dev)))
 
# b) bigram - linear svc
lin_svc_big = LinearSVC(C = .13)
lin_svc_big.fit(X_train_big_aug, y_train)
y_pred_dev = lin_svc_big.predict(X_dev_big_aug)
print("Accuracy on test set of bigram linear svc: {}".format(accuracy_score(y_pred_dev, y_dev)))
 
# c) trigram - linear svc
lin_svc_trig = LinearSVC(C = .13)
lin_svc_trig.fit(X_train_trig_aug, y_train)
y_pred_dev = lin_svc_trig.predict(X_dev_trig_aug)
print("Accuracy on test set of trigram linear svc: {}".format(accuracy_score(y_pred_dev, y_dev)))
 

### Best model
unig_big_vectorizer  = CountVectorizer(stop_words = stop_words, 
                                       ngram_range = (1, 2))
X_train_unig_big     = unig_big_vectorizer.fit_transform(X_train)
X_dev_unig_big       = unig_big_vectorizer.transform(X_dev)
X_train_unig_big_aug = append_extra_feature(X_train_unig_big, append_lst_train)
X_dev_unig_big_aug   = append_extra_feature(X_dev_unig_big, append_lst_dev)

nb_unig_big = MultinomialNB(alpha = .1)
nb_unig_big.fit(X_train_unig_big_aug, y_train)
y_pred_dev = nb_unig_big.predict(X_dev_unig_big_aug)
print("Accuracy on test set of Best model: {}".format(accuracy_score(y_pred_dev, y_dev)))


### Save best model
import pickle as pkl
f_con = open('model.pkl', 'wb')
pkl.dump(nb_unig_big, f_con)
pkl.dump(unig_big_vectorizer, f_con)
f_con.close()

from scipy.sparse import save_npz
save_npz("X_test.npz", X_dev_unig_big_aug)
np.savetxt("y_test.txt", y_dev, delimiter = ',')
