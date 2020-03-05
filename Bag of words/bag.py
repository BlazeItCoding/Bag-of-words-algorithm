import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


phrases = ["The quick brown fox jumped over the lazy dog",
          "education is what you have left after after forgetting everything ever learnt"]

vect = CountVectorizer()
vect.fit(phrases)


print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))

bag_of_words = vect.transform(phrases)

print(bag_of_words)

print("bag_of_words as an array:\n{}".format(bag_of_words.toarray()))

print(vect.get_feature_names())

data = pd.read_csv("C:/Users/marti/Desktop/Artificial Intelligence/labeledTrainData.tsv", delimiter="\t")

print(data)

print(data.head())

#print("Samples per class:{}".format(np.bincount(data.sentiment)))

#def simple_split(data, y, length, split_mark = 0.7):
    #if split_mark > 0. and split_mark < 1.0:
        #n=int(split_mark*length)
    #else:
       # n= int(split_mark)
    #x_train = data[:n].copy()
    #x_test = data[n:].copy()
    #y_train = y[:n].copy()
    #y_test = y[n:].copy()
   # return x_train, x_test, y_train, y_test

