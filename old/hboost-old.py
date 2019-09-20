# #!/usr/bin/env python
# # coding: utf-8
#
# # <a href="https://colab.research.google.com/github/vin-nag/crypto_boost/blob/master/presentation_cv_breast_cancer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # ## Simple Holenstein Boosting Algorithm
# # ### Based on Dr. Russell Impagliazzo's paper
#
# # In[1]:
#
#
# # imports
#
# import math
# import numpy as np
# from numpy.random import choice
# from numba import jit, njit
#
# from sklearn.tree import DecisionTreeClassifier
#
#
# # Aggregating weak learners using simple holenstein boosting algorithm
#
# class HolensteinBoostClassifier:
#
#     def __init__(self, max_depth=1, eps=0.05):
#         """
#         Standard Init function
#         :return: none
#         """
#         self.weak_learners = []
#         self.threshold = 0
#         self.eps = eps
#         HolensteinBoostClassifier.map_to_interval = np.vectorize(HolensteinBoostClassifier.map_to_interval)
#
#
#     @staticmethod
#     @jit(parallel=True)
#     def train_weak_learner(X, y, weights, max_depth=1, probabilities=None):
#         """
#         This function performs one iteration of weak learner training, as described in the dense model paper
#         :param X: nd-array of size m*n training data
#         :param y: 1d-array of size n labels
#         :param weights: 1d-array of size n weights corresponding to weight of each instance
#         :param probabilities: 1d-array of size n corresponding to probabilities for each instance
#         :param indices: 1d-array of size n corresponding to indices of each array
#         :return: weak learner, indices of instances used to fit
#         """
#
#         # get indices of selected samples to train
#         selected_indices = HolensteinBoostClassifier.use_rejection_sample(np.arange(len(weights)), weights, probabilities)
#
#         # select data based on indices
#         X_selected = X[selected_indices, :]
#         y_selected = y[selected_indices]
#
#         # train the weak learner
#         stump = DecisionTreeClassifier(max_depth=max_depth)
#         stump.fit(X_selected, y_selected)
#
#         return stump
#
#
#     @staticmethod
#     @jit(parallel=True)
#     def use_rejection_sample(elements, weights, probabilities=None, size=None, replacement=True):
#         """
#         This function performs rejection sampling, selecting an element based on its probability,
#         and accepting it based on its weight
#         :param elements: 1d-array elements to choose
#         :param probabilities: 1d-array, same size as elements array
#         :param weights: 1d-array, same size as elements array
#         :param size: int, size of array. if None, size defaults to number of elements//10
#         :param replacement: boolean, whether to allow for replacement in set. default=False
#         :return: 1d-array of length size """
#
#         if size is None:
#             size = len(elements) // 10
#
#         if not (probabilities == None):
#             weights *= probabilities
#
#         return choice(elements, p=weights / sum(weights), size=size, replace=replacement)
#
#
#     def reset(self):
#         """
#         This function resets class variables
#         :return: none
#         """
#         self.weak_learners = []
#         self.threshold = 0
#         return
#
#     @staticmethod
#     @jit(parallel=True)
#     def update_sums(predictions, sums, y):
#         """
#         This function updates the weight of each instance based on the performance of all the weak learners on
#         :param predictions: 1d-array of predictions by a weak learner
#         :param sums: 1d-array of current sums of all instances
#         :param y: 1d-array of true labels of values in [1,-1]
#         :return: 1d-array of sums
#         """
#         results = np.multiply(predictions, y)
#         sums = np.add(results, sums)
#         return sums
#
#     @jit(parallel=True)
#     def update_weights(self, sums, weights):
#         """
#         This function updates the weight of each instance based on the performance of all the weak learners on
#         :param weak_learners: 1d-array of trained weak learners that can predict
#         :param X: nd-array of size m*n training data
#         :param y: 1d-array of size n training labels
#         :param weights: 1d-array of size n weights for each instance
#         :param indices: 1d-array of size c<=n indices of instances to predict
#         :param s: int value used to artificially increase density of set
#         :return: 1d-array of weights
#         """
#
#         for index in np.arange(len(weights)):
#             weights[index] =  min(math.exp(-self.eps * sums[index]), 1.0)
#         return weights
#
#     @staticmethod
#     @njit
#     def map_to_interval(num, epsilon=0.1):
#         """
#         This function maps an Integer to the interval [0,1], using the properties described in
#         dense model paper M[num] = min(1, e^(-epsilon*num))
#         :param num: int value to map
#         :param epsilon: float scaling factor
#         :return: float mapped version of num
#         """
#         return
#
#     def fit(self, X, y, n_estimators=100):
#         """
#         This function fits the boosting model to the training data, and hyperparameter tunes the model
#         :param X: nd-array of training instances
#         :param y: 1d-array of labels
#         :param delta: parameter referring to the hard-core size of the set, or alternatively, the accuracy
#         :param epsilon: parameter representing indistinguishability of distributions
#         :param n_estimators: int representing the number of weak learners to train. default of 100
#         :return: none
#         """
#         # reset class level variables
#         self.reset()
#
#         # declare variables
#         counter = 0
#
#         weights = np.array([1 / len(X) for elem in X])
#         sums = [0 for elem in X]
#
#         # repeat C=n_estimator times
#         while counter < n_estimators:
#             # train next weak learner
#             learner = HolensteinBoostClassifier.train_weak_learner(X, y, weights, None)
#
#             # get predictions of trained weak learner
#             predictions = HolensteinBoostClassifier.predict_each_learner(learner, X)
#
#             # update sums of all instances based on predictions
#             sums = HolensteinBoostClassifier.update_sums(predictions, sums, y)
#
#             # update weights of all instances based on sum
#             weights = self.update_weights(sums, weights)
#
#             # add weak learner to ensemble
#             self.weak_learners.append(learner)
#
#             # increment counter
#             counter += 1
#
#         return
#
#     @jit(parallel=True)
#     def sumAllPredictions(self, X):
#         """
#         This function calculates the sum of all instances
#         :param X: nd-array of instances
#         return sums: 1d-array of sums size of length of instances
#         """
#         # declare variable
#         sums = np.zeros(len(X))
#         # loop through all weak learners
#         for learner in self.weak_learners:
#             # add sums for all instances
#             sums = np.add(sums, learner.predict(X))
#         return sums
#
#     @staticmethod
#     @jit(parallel=True)
#     def predict_each_learner(learner, X):
#         """
#         This method predicts an instance, using majority vote
#         :param instance: 1d-array instance to predict
#         :return: int prediction
#         """
#         predictions = [learner.predict(x.reshape(1, -1))[0] for x in X]
#         return predictions
#
#     @jit(parallel=True)
#     def predict(self, X):
#         """
#         This method predicts all instances, using majority vote
#         :param X: 2d-array data to predict
#         :return: 1d-array predictions
#         """
#         return np.sign(self.sumAllPredictions(X))
#
