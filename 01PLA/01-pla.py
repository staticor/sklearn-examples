# -*- coding: utf-8 -*-
# @Author: Jinyong Yang
# @Date:   2019-09-27 08:20:13
# @Last Modified by:   Jinyong Yang
# @Last Modified time: 2019-09-27 08:22:39

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-4, random_state=1428)
clf.fit(X,y)
# Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,penalty=None, random_state=0, shuffle=True, tol=0.001,validation_fraction=0.1, verbose=0, warm_start=False)

print(clf.score(X, y)) # 0.9526989426822482
print(clf.coef_)
print(clf.n_iter_) # 23