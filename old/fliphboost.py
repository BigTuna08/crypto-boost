#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/vin-nag/crypto_boost/blob/master/presentation_cv_breast_cancer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Simple Holenstein Boosting Algorithm
# ### Based on Dr. Russell Impagliazzo's paper

# In[1]:


# imports

import math
import numpy as np
from numpy.random import choice
from numba import jit, njit

from sklearn.tree import DecisionTreeClassifier


# Aggregating weak learners using simple holenstein boosting algorithm

class HolensteinBoostClassifier:

    def __init__(self, max_depth=1, eps=0.05):
        """
        Standard Init function
        :return: none
        """
        self.weak_learners = []
        self.max_depth = max_depth
        self.eps = eps


    def fit(self, X, y, n_estimators=100):
        """
        :param X: nd-array of training instances
        :param y: 1d-array of labels
        :param n_estimators: int representing the number of weak learners to train. default of 100
        :return: none
        """

        n = len(X)
        size = n // 10   # size of subsample for training each tree
        weights = np.full(n, 1/n) # uniform distribution


        self.weak_learners = []
        for i in range(n_estimators):  # create n weak learners


            inds = choice(np.arange(n), p=weights, size=size, replace=True)
            X_selected = X[inds, :]
            y_selected = y[inds]


            stump = DecisionTreeClassifier(max_depth=self.max_depth)
            stump.fit(X_selected, y_selected)


            for i, correct in enumerate(stump.predict(X_selected)*y_selected): #update weights, correct is 1/-1
                self.eps = -self.eps
                weights[i] *= math.exp(-self.eps*correct)

            # np.linalg.norm(weights, ord=1)  # ensure sum(weights) = 1
            weights /= sum(weights)


            self.weak_learners.append(stump)  # add weak learner to ensemble


    def predict(self, X):
        if len(self.weak_learners) < 1:
            raise Exception("Must fit before predicting!")

        return np.sign(sum(map(lambda learner: learner.predict(X), self.weak_learners)))


    # accuracy
    def score(self, X, y):
        return sum(y==self.predict(X))/len(y)



