#!/usr/bin/env python
# coding: utf-8

# ## Simple Holenstein Boosting Algorithm
# ### Based on Dr. Russell Impagliazzo's paper


# imports
import numpy as np
from matplotlib import pyplot as plt

import timeit
from tqdm.auto import tqdm
from numba import jit, njit

# ml related
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_openml

from fhboost import HolensteinBoostClassifier as Fboost
from invhboost import HolensteinBoostClassifier as Invboost
from randhboost import HolensteinBoostClassifier as Rbboost
from randhboost2 import HolensteinBoostClassifier as R2boost
from fliphboost import HolensteinBoostClassifier as Flboost
from enboost import HolensteinBoostClassifier as EnBoost




#################### Helper functions ####################################

def convert_labels(x):
    return -1 if x == 0 else x

convert_labels = np.vectorize(convert_labels, otypes=[np.float])




###################        Params          ###########################################

NUM_ESTIMATORS = 100
FOLDS = 10
MAX_TREE_DEPTH = 2

#################         Data Set         ##################################################
# Pick (uncomment) one

# breast cancer
features, labels = load_breast_cancer(return_X_y=True)
labels = convert_labels(labels)
ds_name = "Breast Cancer"

# blood transfusion
# data = fetch_openml(name='blood-transfusion-service-center')
# features = data.data
# labels = np.array([1 if x=="1" else -1 for x in data.target])
# ds_name = "Blood Transfusion"
#


############### Main process of the algorithm      ############################

print('features shape = ', features.shape)
print('labels shape = ', labels.shape)


times = []
all_scores = []
classifiers = [Fboost(base_estimator=DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH)),
               # Invboost(max_depth=MAX_TREE_DEPTH),
               # Rbboost(max_depth=MAX_TREE_DEPTH),
               # R2boost(max_depth=MAX_TREE_DEPTH),
               #  Flboost(max_depth=MAX_TREE_DEPTH),
               EnBoost(max_depth=MAX_TREE_DEPTH),
               AdaBoostClassifier(n_estimators=NUM_ESTIMATORS,
                                  base_estimator=DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH),
                                  algorithm='SAMME'),
               AdaBoostClassifier(n_estimators=NUM_ESTIMATORS,
                                  base_estimator=DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH),
                                  algorithm='SAMME.R'),
               GradientBoostingClassifier(n_estimators=NUM_ESTIMATORS, max_depth=MAX_TREE_DEPTH)
               ]

boost_names = ['Hboost',
               # 'invhboost',
               # 'randhboost',
               # 'randhboost2',
               # 'fliphboost',
               "ENGER",
               'AdaBoostD',
                'AdaBoostR',
               'Gradient Descent Boost']



skf = StratifiedKFold(n_splits=FOLDS, shuffle=True)

for clf_i, clf in enumerate(classifiers):
    s_time = timeit.default_timer()
    scores = []

    # pbar = tqdm(total=10 * FOLDS)
    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf.fit(X_train, y_train)
        scores.append(clf.score(X=X_test, y=y_test))
        # pbar.update(10)

    times.append(timeit.default_timer() - s_time)
    all_scores.append(scores)




####################  Figures   ########################################
id = None  # for creating images
with open("id") as f:
    id = int(f.readline().strip()) + 1

with open("id", "w") as f:
    print("id is", id)
    f.write(str(id))


# combine the three scores for box-plots

all_scores2 = np.stack((all_scores[0], all_scores[1], all_scores[2]), axis=1)
al_s = all_scores
all_scores = np.transpose(np.array([np.array(sc) for sc in all_scores]))

# Create box plot for the three
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Cross-Validation Performance of boosting models on ' + ds_name + ' Dataset', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=16)

ax.boxplot(all_scores, labels=boost_names)
ax.set_xticklabels(boost_names, rotation=45, fontsize=16)
plt.savefig("acc" + str(id) + ".png")


# plot time taken
tt =times
times2 = np.stack((times[0], times[1], times[2]), axis=0)
times = np.array(times)


fig, ax = plt.subplots(figsize=(10, 10))
index = np.arange(times.size)

# Create box plot for the three
ax.set_ylabel('Time (seconds)', fontsize=16)
ax.set_title('Times taken by the boosting models', fontsize=20)

ax.bar(index, times, alpha=0.4, color='b', label='Times')

ax.set_xticks(index)
ax.set_xticklabels(boost_names, rotation=45, fontsize=16)

plt.savefig("time" + str(id) + ".png")
