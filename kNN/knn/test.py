from __future__ import print_function
import sys, os
sys.path.append("../../")
sys.path.append(".")

from keras.preprocessing import sequence
from training.keras.dataset import ArtrificialDataset, GaussianGenerator
from emails import get_emails

from queue import PriorityQueue

import knn
import distance

if __name__ == "__main__":
    output_file = "knn_results.csv"
    emails = get_emails("emails.txt")
    dataset = ArtrificialDataset("training_data", GaussianGenerator(), 1)

    for mail in ["kobojekp@student.mini.pw.edu.pl"]:
        X_data, Y_data = dataset.load(mail)
        print("X_data", X_data)
        print("Y_data", Y_data)
