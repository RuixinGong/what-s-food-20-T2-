# Author:  WenJie Yu

import json
import time
import matplotlib
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
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

accuracy_list = []
parameter_list = []
for parameter in range(2, 15):

    clf_2 = make_pipeline(TfidfVectorizer(), KNN(n_neighbors=13,
                                                 weights='distance',
                                                 leaf_size=30,
                                                 p=2,))

    start_time = time.time()
    scores = cross_validate(clf_2, ingredients_list,
                            cuisine_list,  cv=5)

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
plt.xlabel('n_neighbors from 2 to 15, weights = distance')
plt.ylabel('accuracy')
plt.show()
