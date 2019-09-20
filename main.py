#!/usr/bin/env python
# coding: utf-8

# ## Simple Holenstein Boosting Algorithm
# ### Based on Dr. Russell Impagliazzo's paper


# imports
import numpy as np
from my_plotting import plt_acc, plt_scores, plt_acc_colored

import timeit

from helpers import get_data

# ml related

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold


from fhboost import HolensteinBoostClassifier as Fboost
from fhboost import InfoHBoost
# from invhboost import HolensteinBoostClassifier as Invboost
# from randhboost import HolensteinBoostClassifier as Rbboost
# from randhboost2 import HolensteinBoostClassifier as R2boost
# from fliphboost import HolensteinBoostClassifier as Flboost
# from enboost import HolensteinBoostClassifier as EnBoost




###################        Params          ###########################################

NUM_ESTIMATORS = 100
FOLDS = 10
MAX_TREE_DEPTH = 2
REPS = 5


# noise_levels = [0.0, 0.1, 0.2]
noise_levels = [0.1*i for i in range(6)]
noise_type = "feat"  # "lbl" or "feat"

epss = [0.05*i for i in range(1,6)]
# delts = [0.025*i for i in range(4,13)]
# delts = [0.025*i for i in range(8,10)]
delts = [0.05*i for i in range(2,10)] #+ [0.025*i for i in range(8,13)]

#################         Data Set         ##################################################
data_sets = ["Breast Cancer",
             "Blood Transfusion"]
ds_name = data_sets[0]

true_features, true_labels = get_data(data_sets[0])
features, labels = true_features.copy(), true_labels.copy()


##############       Set classifiers     ##############################################################

classifiers = [
    # Fboost(base_estimator=DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH)),
               # Invboost(max_depth=MAX_TREE_DEPTH),
               # Rbboost(max_depth=MAX_TREE_DEPTH),
               # R2boost(max_depth=MAX_TREE_DEPTH),
               #  Flboost(max_depth=MAX_TREE_DEPTH),
               AdaBoostClassifier(n_estimators=NUM_ESTIMATORS,
                                  base_estimator=DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH),
                                  algorithm='SAMME'),
               GradientBoostingClassifier(n_estimators=NUM_ESTIMATORS, max_depth=MAX_TREE_DEPTH)
               ]

boost_names = [
    # 'Hboost',
               # 'invhboost',
               # 'randhboost',
               # 'randhboost2',
               # 'fliphboost',
               'AdaBoostD',
               'Gradient Boost']

for eps in epss:
    for delta in delts:
        # classifiers.append(Fboost(base_estimator=DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH),
        #                           eps=eps,
        #                           delta=delta))
        # boost_names.append("H|e={:.2f},d={:.2f}".format(eps, delta))

        classifiers.append(InfoHBoost(base_estimator=DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH),
                                  eps=eps,
                                  delta=delta))
        boost_names.append("I|e={:.2f},d={:.2f}".format(eps, delta))



if __name__ == '__main__':

    for noise_level in noise_levels:
        print("noise level:", noise_level , "\n\n\n\n\n")

        if noise_type == "lbl":
            labels = true_labels.copy()
            flip_lbls = np.random.choice(np.arange(len(true_labels)), size=int(len(true_labels) * noise_level),
                                         replace=False)
            labels[flip_lbls] = -labels[flip_lbls]
            ds_name = "{} with label noise={}".format(ds_name, noise_level)
        elif noise_type == "feat":
            features = np.multiply( np.random.normal(1, noise_level, true_features.shape), true_features)
            ds_name = "{} with feature noise={}".format(ds_name, noise_level)
        elif noise_type is None:
            pass
        else:
            raise Exception("Invalid noise type")

        times = []
        all_scores = []

        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True)

        for clf_i, clf in enumerate(classifiers):
            s_time = timeit.default_timer()
            scores = []


            for train_index, test_index in skf.split(features, labels):
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                clf.fit(X_train, y_train)
                scores.append(clf.score(X=X_test, y=y_test))

            times.append(timeit.default_timer() - s_time)
            all_scores.append(scores)


        ####################  Figures   ########################################
        id = None  # for creating images
        with open("logs/id") as f:
            id = int(f.readline().strip()) + 1

        with open("logs/id", "w") as f:
            print("id is", id)
            f.write(str(id))

        # plt_acc_colored(all_scores, ds_name, boost_names, noise_level, id)
        plt_acc(all_scores, ds_name, boost_names, noise_level, id)

