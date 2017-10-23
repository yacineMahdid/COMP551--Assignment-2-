from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import re
import math
import numpy as np
import pandas as pd
import string
import operator
from sklearn.decomposition import PCA
import time
from tqdm import *

#translate table for punctuations and emoji characters initialized
translate_table = dict((ord(char), None) for char in string.punctuation)
emoji_pattern = re.compile(u"[^\U00000000-\U0000d7ff-\U0000e000-\U0000ffff]",flags=re.UNICODE);


def cleaning_and_vectorization(dataframe):
    #cleaning data frame
    for i, line in dataframe.iterrows():
        row = line['Text']
        row = re.sub(r'(\s)http\w+', '', row)
        row = ''.join([i for i in row if not i.isdigit()])
        row = row.lower()
        if len(row) != 0:
            row = row.translate(translate_table)
        for i in row:
            re.sub(' +', ' ', i)
        row = emoji_pattern.sub(r'', row)
        line['Text'] = row

    for i, line in dataframe.iterrows():
        try:
            words = line['Text']
            words_split = ''.join(words.split())
            train_data.set_value(i, 'Text', words_split)
        except:
            pass
    # Tf IDF based on character created
    tf = TfidfVectorizer(analyzer='char', min_df=12)
    tfidf_matrix = tf.fit_transform(dataframe['Text'])
    # converted into a dense matrix
    tfidf_dense = tfidf_matrix.todense()
    # Principal component analysis selects 50 features
    pca = PCA(n_components=50)
    tfidf_final = pca.fit_transform(tfidf_dense)
    if 'Category' in dataframe:
        #training data matrix created
        Y = dataframe['Category']
        final_corpus = np.column_stack((tfidf_final,Y.T))
    else:
        #testing data matrix created
        final_corpus = tfidf_final
    return final_corpus



def euclideanDistance(instance_x, instance_y, length):
    distance = 0
    for x in range(length):
        distance += pow((instance_x[x] - instance_y[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(trainingSet.shape[0]):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours

def getResponse(neighbours):
    Votes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in Votes:
            Votes[response] += 1
        else:
            Votes[response] = 1
    sortedVotes = sorted(Votes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

#accuracy function was used on the subsetted training data in cross validation
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

predictions = []
k = 3


# Reading training and testing data
train_data = pd.read_csv("train_set_x.csv")
test_data = pd.read_csv("test_set_x.csv")

#matrix representations created
final_training = cleaning_and_vectorization(train_data)
final_testing = cleaning_and_vectorization(test_data)

#kNN lazy learning model created based on the testing instances
for x in tqdm(range(final_testing.shape[0])):
    time.sleep(3)
    neighbors = getNeighbors(final_training, final_testing[x], k)
    result = getResponse(neighbors)
    result_dict = dict(id = x, Category = result)
    predictions.append(result_dict)

keys = predictions[0].keys()

#predictions written to a csv file
with open('knn_results.csv', 'w', encoding='utf8', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(predictions)
