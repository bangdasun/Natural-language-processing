# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:14:12 2017

@author: Bangda

To run this file, type in terminal or cmd:
    python analyze.py model.pkl X_test.npz y_test.txt

Take a model and print the confusion matrix and top 20 features (most predictive, based on coefficients)
"""

import sys
import pickle as pkl

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix

### load data
X_dev_unig_big = load_npz(sys.argv[2])

with open(sys.argv[3], encoding = 'latin-1') as dev_con:
    y_dev = np.loadtxt(dev_con, delimiter = ',')

### Modeling
f_con = open(sys.argv[1], 'rb')
model = pkl.load(f_con)
vectorizer = pkl.load(f_con)
f_con.close()

feature_names = vectorizer.get_feature_names()
feature_names.append('num_word_lengthening')
feature_names.append('num_all_upper_case')
feature_names.append('num_multi_punct')
feature_names.append('num_hashtag')
feature_names.append('num_emoji')
idx_feature_names = np.argsort(model.coef_[0])[-20:]

y_pred_dev = model.predict(X_dev_unig_big)
print("Confusion matrix of test set:\n {}".format(confusion_matrix(y_dev, y_pred_dev)))
print("Top 20 features:\n {}".format([feature_names[idx] for idx in idx_feature_names]))
