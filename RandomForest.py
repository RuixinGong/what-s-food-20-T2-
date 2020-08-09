# Author: ChenXu Wang

import json
import time
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

with open("train_postprecessing.json") as f:
    data_set = json.load(f)
cuisine_list = []
ingredients_list = []
for element in data_set:
    # id_list.append(element['id'])
    cuisine_list.append(element['cuisine'])
    ingredients_list.append(element['ingredients'])
assert(len(cuisine_list) == len(ingredients_list))

# DecisionTree
accuracy_list = []
parameter_list = []
for parameter in range(1, 21):

    # clf_1 = make_pipeline(TfidfVectorizer(max_features=1500),
    # tree(
    # random_state=0,
    # splitter='random',

    #clf = make_pipeline(TfidfVectorizer(), tree(criterion="gini", max_depth=50,min_samples_leaf=10, splitter="random"))

    clf_2 = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=32,
                                                                    criterion='gini',
                                                                    max_depth=65,
                                                                    min_samples_split=2,
                                                                    min_samples_leaf=parameter,
                                                                    min_weight_fraction_leaf=0.0,
                                                                    max_features='auto',
                                                                    max_leaf_nodes=None,
                                                                    min_impurity_decrease=0.0,
                                                                    min_impurity_split=None,
                                                                    bootstrap=True,
                                                                    oob_score=False,
                                                                    n_jobs=1,
                                                                    random_state=None,
                                                                    verbose=0,
                                                                    warm_start=False,
                                                                    class_weight=None))

    start_time = time.time()
    scores = cross_validate(clf_2, ingredients_list,
                            cuisine_list,  cv=5)  # scoring=['accuracy'],
    accuracy = scores['test_score'].mean()
    accuracy_list.append(accuracy)
    parameter_list.append(parameter)
    print(accuracy)
    print("Running time:  %s seconds" % (time.time() - start_time))


# plot graph
matplotlib.rc('axes', facecolor='white')
matplotlib.rc('figure', figsize=(6, 4))
matplotlib.rc('axes', grid=False)

plt.plot(parameter_list, accuracy_list, '*:r')

plt.title('')
plt.xlabel('min_samples_leaf')
plt.ylabel('accuracy')
plt.show()
