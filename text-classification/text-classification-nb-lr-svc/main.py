# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:15:04 2017

@author: Bangda
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:38:50 2017

@author: Bangda
"""

import os

os.system("python classify.py train.txt dev.txt")
print("\n\n********************Unigram Model********************\n\n")
os.system("python analyze.py unig-nb.pkl X_test_unig.npz y_test.txt")
print("\n\n********************Bigram Model********************\n\n")
os.system("python analyze.py big-lr.pkl X_test_big.npz y_test.txt")
print("\n\n********************Trigram Model********************\n\n")
os.system("python analyze.py trig-svc.pkl X_test_trig.npz y_test.txt")
print("\n\n********************Best Model********************\n\n")
os.system("python analyze.py model.pkl X_test_best.npz y_test.txt")