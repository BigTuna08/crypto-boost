import math
import numpy as np
from numpy.random import choice
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial import distance







# more Holenstein like

class HolensteinBoostClassifier:

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50,
                 eps=0.1,
                 delta=0.2,
                 subsample=0.5,
                 weight_updater=None):
        """

        """
        self.weak_learners = []
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.eps = eps
        self.delta = delta
        self.subsample = subsample

        self.weight_updater = weight_updater
        if not self.weight_updater:
            self.weight_updater = self.default_wu


    def fit(self, X, y):

        n = len(X)
        s = 0
        sample_size = math.ceil(n * self.subsample)
        advantages = np.full(n, 0.)
        weights = np.full(n, 1)  # sample weight in the measure
        self.weak_learners = []
        initial_distr = np.full(n,1/n)

        for _ in range(self.n_estimators):  # create n weak learners

            # select subsample to train current weak learner on
            induced_distr = weights.copy() / sum(weights)  # assumes base distr is uniform
            inds = choice(np.arange(n), p=induced_distr, size=sample_size, replace=True)
            X_selected, y_selected = X[inds, :], y[inds]

            # fit new WL
            wl = self.base_estimator.fit(X_selected, y_selected)

            # use new WL for prediction
            classification_result = wl.predict(X) * y  # vector, +1 if correct, -1 otherwise
            advantages += classification_result

            # update weights
            weights = np.array([min(1, math.exp(-self.eps*(adv-s))) for adv in advantages])


            density = sum(weights) / len(weights)  # assumes base distr is uniform
            if density < 2 * self.delta:
                s = s + 1

            self.weak_learners.append(wl)  # add weak learner to ensemble


        with open("log", "a") as f:
            f.write("{} / {} ={:.3f}| eps={:.3f}, delta={:.3f}\n".format(s, self.n_estimators, s/self.n_estimators, self.eps, self.delta))



    def default_wu(self, current_weights, classification_result, s):
        for i, correct in enumerate(classification_result):  # update weights
            current_weights[i] *= math.exp(-self.eps * (correct - s))
            current_weights[i] = min(current_weights[i], 1)

    def predict(self, X):
        if len(self.weak_learners) < 1:
            raise Exception("Must fit before predicting!")

        return np.sign(sum(map(lambda learner: learner.predict(X), self.weak_learners)))

    # accuracy
    def score(self, X, y):
        return sum(y == self.predict(X)) / len(y)



class InfoHBoost():
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50,
                 eps=0.1,
                 delta=0.2,
                 subsample=0.5,
                 weight_updater=None):
        """

        """
        self.weak_learners = []
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.eps = eps
        self.delta = delta
        self.subsample = subsample

        self.weight_updater = weight_updater
        if not self.weight_updater:
            self.weight_updater = self.default_wu


    def fit(self, X, y):

        n = len(X)
        s = 0
        sample_size = math.ceil(n * self.subsample)
        advantages = np.full(n, 0.)
        weights = np.full(n, 1)  # sample weight in the measure
        self.weak_learners = []
        initial_distr = np.full(n, 1 / n)

        max_dist = 0.0

        for _ in range(self.n_estimators):  # create n weak learners

            # select subsample to train current weak learner on
            induced_distr = weights.copy() / sum(weights)  # assumes base distr is uniform
            inds = choice(np.arange(n), p=induced_distr, size=sample_size, replace=True)
            X_selected, y_selected = X[inds, :], y[inds]

            # fit new WL
            wl = self.base_estimator.fit(X_selected, y_selected)

            # use new WL for prediction
            classification_result = wl.predict(X) * y  # vector, +1 if correct, -1 otherwise
            advantages += classification_result

            # update weights
            weights = np.array([min(1, math.exp(-self.eps * (adv - s))) for adv in advantages])
            weights /= sum(weights)

            dist = distance.jensenshannon(weights, initial_distr)
            max_dist = max(max_dist, dist)
            print("dist: ", dist, "with e,d=", self.eps, self.delta)

            # if distance.jensenshannon(weights, initial_distr) >1- 2*self.delta: # if to far from uniform
            #     s = s + 1

            if dist > 1/3 * (1-2*self.delta): # if to far from uniform
                s = s + 1

            self.weak_learners.append(wl)  # add weak learner to ensemble

        with open("log", "a") as f:
            f.write("I{} / {} ={:.3f}|   maxd={}   |    eps={:.3f}, delta={:.3f}\n".format(s, self.n_estimators, s / self.n_estimators, max_dist,
                                                                         self.eps, self.delta))


    def default_wu(self, current_weights, classification_result, s):
        for i, correct in enumerate(classification_result):  # update weights
            current_weights[i] *= math.exp(-self.eps * (correct - s))
            current_weights[i] = min(current_weights[i], 1)

    def predict(self, X):
        if len(self.weak_learners) < 1:
            raise Exception("Must fit before predicting!")

        return np.sign(sum(map(lambda learner: learner.predict(X), self.weak_learners)))

    # accuracy
    def score(self, X, y):
        return sum(y == self.predict(X)) / len(y)



########################## I think good ################

#
#
# class HolensteinBoostClassifier:
#
#     def __init__(self,
#                  base_estimator=DecisionTreeClassifier(max_depth=1),
#                  n_estimators=50,
#                  eps=0.1,
#                  delta=0.2,
#                  subsample=0.5,
#                  weight_updater=None):
#         """
#
#         """
#         self.weak_learners = []
#         self.base_estimator = base_estimator
#         self.n_estimators = n_estimators
#         self.eps = eps
#         self.delta = delta
#         self.subsample = subsample
#
#         self.weight_updater = weight_updater
#         if not self.weight_updater:
#             self.weight_updater = self.default_wu
#
#
#     def fit(self, X, y):
#         """
#         """
#
#         n = len(X)
#         s = 0
#         sample_size = math.ceil( n *self.subsample)
#         weights = np.full(n, 1/ n)  # uniform distribution
#         initial_distr = np.full(n, 1 / n)  # uniform distribution
#         self.weak_learners = []
#
#         for _ in range(self.n_estimators):  # create n weak learners
#
#             # select subsample to train current weak learner on
#             inds = choice(np.arange(n), p=weights, size=sample_size, replace=True)
#             X_selected, y_selected = X[inds, :], y[inds]
#
#             # fit base estimator
#             stump = self.base_estimator.fit(X_selected, y_selected)
#
#             # update weights
#             classification_result = stump.predict(X_selected) * y_selected  # vector, +1 if correct, -1 otherwise
#             self.weight_updater(weights, classification_result, s)
#
#             print("dist is", distance.jensenshannon(weights, initial_distr))
#             if distance.jensenshannon(weights, initial_distr) < 1 - self.delta:
#                 s = s + 1
#                 print("        Updated s       ")
#
#             weights /= sum(weights)  # normalize, sum(weights) = 1
#
#             self.weak_learners.append(stump)  # add weak learner to ensemble
#
#     def default_wu(self, current_weights, classification_result, s):
#         for i, correct in enumerate(classification_result):  # update weights
#             current_weights[i] *= math.exp(-self.eps * (correct - s))
#             current_weights[i] = min(current_weights[i], 1)
#
#     def predict(self, X):
#         if len(self.weak_learners) < 1:
#             raise Exception("Must fit before predicting!")
#
#         return np.sign(sum(map(lambda learner: learner.predict(X), self.weak_learners)))
#
#     # accuracy
#     def score(self, X, y):
#         return sum(y == self.predict(X)) / len(y)
#
#
#
#

















#########   Old #######################









# #!/usr/bin/env python
# # coding: utf-8
#
#
# # ## Simple Holenstein Boosting Algorithm
# # ### Based on Dr. Russell Impagliazzo's paper
#
#
# import math
# import numpy as np
# from numpy.random import choice
#
# from sklearn.tree import DecisionTreeClassifier
#
#
# # Aggregating weak learners using simple holenstein boosting algorithm
#
# class HolensteinBoostClassifier:
#
#     def __init__(self, max_depth=1, eps=0.1):
#         """
#         Standard Init function
#         :return: none
#         """
#         self.weak_learners = []
#         self.max_depth = max_depth
#         self.eps = eps
#
#
#     def fit(self, X, y, n_estimators=100):
#         """
#         :param X: nd-array of training instances
#         :param y: 1d-array of labels
#         :param n_estimators: int representing the number of weak learners to train. default of 100
#         :return: none
#         """
#
#         n = len(X)
#         # size = n // 10   # size of subsample for training each tree
#         size = n
#         weights = np.full(n, 1/n) # uniform distribution
#
#
#         self.weak_learners = []
#         for _ in range(n_estimators):  # create n weak learners
#
#
#             inds = choice(np.arange(n), p=weights, size=size, replace=True)
#             X_selected = X[inds, :]
#             y_selected = y[inds]
#
#
#             stump = DecisionTreeClassifier(max_depth=self.max_depth)
#             stump.fit(X_selected, y_selected)
#
#
#             for i, correct in enumerate(stump.predict(X_selected)*y_selected): #update weights, correct is 1/-1
#                 weights[i] *= math.exp(-self.eps*correct)
#                 weights[i] = min(weights[i], 1)
#             weights /= sum(weights)   # ensure sum(weights) = 1
#
#
#             self.weak_learners.append(stump)  # add weak learner to ensemble
#
#
#     def predict(self, X):
#         if len(self.weak_learners) < 1:
#             raise Exception("Must fit before predicting!")
#
#         return np.sign(sum(map(lambda learner: learner.predict(X), self.weak_learners)))
#
#
#     # accuracy
#     def score(self, X, y):
#         return sum(y==self.predict(X))/len(y)
