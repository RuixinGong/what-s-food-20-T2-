# Author: LiRen Xu

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

############ CHANGE HERE ##################
clf = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=1, fit_prior=False))
###########################################

start_time = time.time()
scores = cross_validate(clf, ingredients_list,
                        cuisine_list, scoring=['accuracy'], cv=5)
print(scores)
print("Running time:  %s seconds" % (time.time() - start_time))


# plot graph
x = np.linspace(0.1, 3.0, 30)
outputlist = []
for i in x:
    print(i)
    y1 = make_pipeline(TfidfVectorizer(),
                       MultinomialNB(alpha=i, fit_prior=False))
    y2 = cross_validate(y1, ingredients_list,
                        cuisine_list, scoring=['accuracy'], cv=5)
    y = y2['test_accuracy']
    output = 0
    for i in y:
        output = output + i
    output = output/len(y)
    outputlist.append(output)
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.plot(x, outputlist)
plt.show()
