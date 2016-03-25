from __future__ import print_function
import sys, os
sys.path.append("../../")

import numpy as np
from scipy import spatial
from keras.preprocessing import sequence
from training.keras import dataset
from emails import get_emails
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []

    def push(self, priority, item):
        heapq.heappush(self._queue, (-priority, item))

    def pop(self):
        if len(self._queue) == 0:
            return None
        return heapq.heappop(self._queue)[-1]

    def peek_priority(self):
        return -self._queue[0][0]

    def size(self):
        return len(self._queue)

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def max_distance(x, y):
    return np.max(abs(x - y))

def manhattan(x, y):
    return spatial.distance.cityblock(x, y)


def knn_predict(X_data, Y_data, example, distance, k):

    prediction = 0
    pq = PriorityQueue()
    for i in range(len(X_data)):
        row = X_data[i, :]
        d = distance(row, example)
        if pq.size() < k:
            pq.push(d, Y_data[i])
        elif d < pq.peek_priority():
            pq.pop()
            pq.push(d, Y_data[i])

    prediction = pq.pop()
    item = pq.pop()
    while item != None:
        prediction = (prediction + item) / 2.0
        item = pq.pop()

    return round(prediction)

if __name__ == "__main__":
    k = 1
    emails = get_emails("../../emails.txt")
    totalFP = 0
    totalFN = 0
    totalTP = 0
    totalTN = 0

    for mail in emails:
        X_data, Y_data = dataset.load_data(mail)

        falsePositives = 0
        falseNegatives = 0
        truePositives  = 0
        trueNegatives  = 0
        for i in range(len(X_data)):
            tmp       = X_data[i]
            X_data[i] = X_data[0]
            X_data[0] = tmp

            tmp       = Y_data[i]
            Y_data[i] = Y_data[0]
            Y_data[0] = tmp

            X_train = X_data[1:]
            y_train = Y_data[1:]
            X_test  = X_data[0:1]
            y_test  = Y_data[0:1]

            X_train = sequence.pad_sequences(X_train, maxlen=100)
            X_test = sequence.pad_sequences(X_test, maxlen=100)

            predicted_class = knn_predict(X_train, y_train, X_test, manhattan, k)

            if(predicted_class == 0 and y_test[0] == 1):
                falseNegatives += 1
            elif(predicted_class == 1 and y_test[0] == 0):
                falsePositives += 1
            elif(predicted_class == 1 and y_test[0] == 1):
                truePositives += 1
            elif(predicted_class == 0 and y_test[0] == 0):
                trueNegatives += 1

            tmp       = X_data[i]
            X_data[i] = X_data[0]
            X_data[0] = tmp

            tmp       = Y_data[i]
            Y_data[i] = Y_data[0]
            Y_data[0] = tmp

            totalFP += falsePositives
            totalFN += falseNegatives
            totalTP += truePositives
            totalTN += trueNegatives


        result_string = "\n".join(["For: %s FP: %d FN: %d TP: %d TN: %d"])
        result_string = result_string % (mail, falsePositives, falseNegatives,
                        truePositives, trueNegatives)

        print(result_string)

    result_string = "TotalFP: " + str(totalFP) + " TotalFN: " + str(totalFN) + \
                    " TotalTP: " + str(totalTP) + " TotalTN: " + str(totalTN)
    print(result_string)

    print("Total accuracy:",
        float(totalTP + totalTN) / float(totalTP + totalTN + totalFP + totalFN),
        "\n kNN with k =", k)

    print("False positive / total errors:", float(totalFP) / float(totalFP + totalFN))
