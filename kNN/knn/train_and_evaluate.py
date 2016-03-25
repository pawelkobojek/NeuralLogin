from __future__ import print_function
import sys, os
sys.path.append("../../")
sys.path.append(".")

from keras.preprocessing import sequence
from training.keras import dataset
from emails import get_emails

from queue import PriorityQueue

import knn
import distance

if __name__ == "__main__":
    output_file = "knn_results.csv"
    emails = get_emails("emails.txt")

    with open(output_file, "w+") as f:
        f.write(",".join(["k", "distance", "TP", "TN", "FP", "FN", "Acc"]))

    for k in range(1, 5):
        for distance_function in distance.all_distances():
            totalFP = 0
            totalFN = 0
            totalTP = 0
            totalTN = 0

            for mail in emails:
                X_data, Y_data = dataset.load_data(mail, "training_data")

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

                    predicted_class = knn.predict(X_train, y_train, X_test,
                                                    distance_function.compute, k)

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

            acc = float(totalTP + totalTN) / float(totalTP + totalTN + totalFP + totalFN)

            print("Total accuracy:",
                acc,
                "\n kNN with k =", k)

            with open(output_file, "a+") as f:
                f.write("\n")
                f.write(",".join([str(k), distance_function.name(),
                                    str(totalTP), str(totalTN), str(totalFP),
                                    str(totalFN), str(acc)]))

            print("False positive / total errors:", float(totalFP) / float(totalFP + totalFN))
