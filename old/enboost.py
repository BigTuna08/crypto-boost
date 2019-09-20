#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/vin-nag/crypto_boost/blob/master/presentation_cv_breast_cancer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Simple Holenstein Boosting Algorithm
# ### Based on Dr. Russell Impagliazzo's paper

# In[1]:


# imports

import math
import numpy as np
from numpy.random import choice, random
from numba import jit, njit

from sklearn.tree import DecisionTreeClassifier


# Aggregating weak learners using simple holenstein boosting algorithm


# l is list of numbers in {1,-1}. converts to bin 2 dec, treating -1 as 0
def l2k(l):
    sum = 0
    for i, item in enumerate(l):
        if item == 1:
            sum += 2**i
    return sum


class HolensteinBoostClassifier:

    def __init__(self, max_depth=1, eps=0.1):
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


        blocks = []
        for _ in range(len(X)):
            blocks.append([])
        self.weak_learners = []


        for iteration in range(n_estimators):  # create n weak learners



            #
            # inds = choice(np.arange(n), p=weights, size=size, replace=True)
            # X_selected = X[inds, :]
            # y_selected = y[inds]

            block_members = {}
            # block_members.setdefault()

            stump = DecisionTreeClassifier(max_depth=self.max_depth)
            stump.fit(X, y, sample_weight=weights)
            res = stump.predict(X)

            for i in range(len(res)):
                blocks[i].append(res[i])
                dk = l2k(blocks[i])

                # if dk in block_members:
                try:
                    block_members[dk].append(y[i])
                except:
                    block_members[dk] = [y[i]]

            for key in block_members:
                pos = block_members[key].count(1)
                neg = block_members[key].count(-1)

                # pw = pos / (pos+neg)
                # nw = neg / (pos + neg)
                # block_members[key] = (pw,nw)


                if pos > neg or (pos == neg and random() > 0.5):
                    maj_bit = 1
                    delta = neg / (pos + neg)
                else:
                    maj_bit = -1
                    delta = pos / (pos + neg)

                # print("*****************delta, p , n", delta, pos, neg)

                block_members[key] = (maj_bit, delta)


            for i in range(len(res)):
                maj_bit, delta = block_members[l2k(blocks[i])]
                # print(maj_bit, y[i], delta)
                if y[i] == maj_bit:
                    weights[i] = delta/(1-delta)
                else:
                    weights[i] = 1
                # print(weights[i])


            # for i, correct in enumerate(stump.predict(X_selected)*y_selected): #update weights, correct is 1/-1
            #     weights[i] *= math.exp(-self.eps*correct)
            # weights /= sum(weights)   # ensure sum(weights) = 1
            if sum(weights) == 0:
                print("Finshed early!!!", iteration)
                break


            weights /= sum(weights)  # ensure sum(weights) = 1


            self.weak_learners.append(stump)  # add weak learner to ensemble


    def predict(self, X):
        if len(self.weak_learners) < 1:
            raise Exception("Must fit before predicting!")

        return np.sign(sum(map(lambda learner: learner.predict(X), self.weak_learners)))


    # accuracy
    def score(self, X, y):
        return sum(y==self.predict(X))/len(y)



