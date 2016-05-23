from __future__ import print_function
import sys, os
import numpy as np
sys.path.append("../../")
sys.path.append(".")

from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

from training.keras.dataset import BenchmarkDataset, GaussianGenerator
from emails import get_emails

from queue import PriorityQueue

import knn
import distance

def train_test_split(X, y, test_size=0.2):
    positive_count = sum(y)

    negative_ys = y[positive_count:]
    negative_Xs = X[positive_count:]

    negative_Xs, negative_ys = shuffle(negative_Xs, negative_ys)
    X = np.concatenate( (X[:positive_count], negative_Xs[:positive_count]) )
    y = np.concatenate( (y[:positive_count], negative_ys[:positive_count]) )

    positive_split_point = int((1.0 - test_size) * positive_count)

    X_train = np.concatenate( (X[:positive_split_point], X[positive_count:positive_count + positive_split_point]) )
    X_test = np.concatenate( (X[positive_split_point:positive_count], X[positive_count + positive_split_point:]) )
    y_train = np.concatenate( (y[:positive_split_point], y[positive_count:positive_count + positive_split_point]) )
    y_test = np.concatenate( (y[positive_split_point:positive_count], y[positive_count + positive_split_point:]) )

    return X_train, X_test, y_train, y_test


def find_best_parameters(output_file):
    with open(output_file, "r") as f:
        results = [d.split(',') for d in f.read().splitlines()[1:]]

    maximum = [0.0]
    for row in results:
        if row[-1] > maximum[-1]:
            maximum = row
    return maximum

if __name__ == "__main__":
    output_file = "benchmark_knn_results.csv"
    dataset_dir = "benchmark_set"
    emails = list(get_emails("subjects.txt"))

    with open(output_file, "w+") as f:
        f.write(",".join(["k", "distance", "std", "TP", "TN", "FP", "FN", "Acc"]))

    for std in [350]:
        dataset = BenchmarkDataset(dataset_dir)
        generator = GaussianGenerator(mean=0, std=std)

        for k in range(1, 5):
            print("k=", k)
            for distance_function in distance.all_distances():
                print("distance:", distance_function.name())
                totalFP = 0
                totalFN = 0
                totalTP = 0
                totalTN = 0
                print(emails)
                for mail in emails:
                    print("subject:", mail)
                    falsePositives = 0
                    falseNegatives = 0
                    truePositives  = 0
                    trueNegatives  = 0

                    X, y = dataset.load(mail)
                    X = np.array(X)
                    y = np.array(y)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    halfind = len(X_train) / 2
                    print(halfind)
                    for i in range(int(halfind)):
                        X_train[halfind + i] = generator.generate(X_train[i])

                    X_train = normalize(X_train)
                    X_test = normalize(X_test)

                    i = 0
                    for sample in X_test:
                        predicted_class = knn.predict(X_train, y_train, sample,
                                                    distance_function.compute, k)
                        if(predicted_class == 0 and y_test[i] == 1):
                            falseNegatives += 1
                        elif(predicted_class == 1 and y_test[i] == 0):
                            falsePositives += 1
                        elif(predicted_class == 1 and y_test[i] == 1):
                            truePositives += 1
                        elif(predicted_class == 0 and y_test[i] == 0):
                            trueNegatives += 1
                        i += 1

                    result_string = "\n".join(["For: %s FP: %d FN: %d TP: %d TN: %d"])
                    result_string = result_string % (mail, falsePositives, falseNegatives,
                                    truePositives, trueNegatives)

                    totalFP += falsePositives
                    totalFN += falseNegatives
                    totalTP += truePositives
                    totalTN += trueNegatives
                    print(result_string)

                result_string = "TotalFP: " + str(totalFP) + " TotalFN: " + str(totalFN) + \
                                " TotalTP: " + str(totalTP) + " TotalTN: " + str(totalTN)
                print(result_string)

                acc = float(totalTP + totalTN) / float(totalTP + totalTN + totalFP + totalFN)

                print("Total accuracy:",
                    acc,
                    "\n kNN with k =", k)

                with open(output_file, "a+") as f:
                    f.write("\n")
                    f.write(",".join([str(k), distance_function.name(), str(std),
                                        str(totalTP), str(totalTN), str(totalFP),
                                        str(totalFN), str(acc)]))

                print("False positive / total errors:", float(totalFP) / float(totalFP + totalFN))

    print("Best parameters found:", find_best_parameters(output_file))
