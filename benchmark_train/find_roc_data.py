import sys, os
from sklearn.cross_validation import train_test_split as splitter
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
sys.path.append(".")

from emails import get_emails

import numpy as np
import theano
np.random.seed(1337)  # for reproducibility

from training.keras.dataset import BenchmarkDataset
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import model_from_json

from config import LSTM3Layers3DropoutsConfig, GRU3Layers3DropoutsConfig, LSTM2Layers2DropoutsConfig

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

def load_model(model_file, model_weights_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(model_weights_file)
    return model

if __name__ == "__main__":
    dataset_dir = "benchmark_set"
    emails_file = "subjects.txt"
    base_results_dir = "benchmark_results"

    dataset = BenchmarkDataset(dataset_dir)
    subjects = get_emails(emails_file)

    configs = ["lstm3layer3dropout", "gru3layer3dropout", "lstm2layer2dropout"]

    for config in configs:
        for subject in subjects:
            model_file = os.path.join(base_results_dir, config, subject, "model.json")
            model_weights_file = os.path.join(base_results_dir, config, subject, "weights.h5")
            out_file = os.path.join(base_results_dir, config, "raw_results", subject + ".txt")

            if not os.path.exists(os.path.dirname(out_file)):
                os.makedirs(os.path.dirname(out_file))

            X, y = dataset.load(subject)
            X = np.array(X)
            y = np.array(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            X_test = normalize(X_test)

            model = load_model(model_file=model_file, model_weights_file=model_weights_file)
            model.compile(loss='binary_crossentropy', optimizer='adam')

            X_test = sequence.pad_sequences(X_test, maxlen=10, dtype='float32')
            X_test = np.reshape(X_test, (X_test.shape[0], 10, 1))

            # Find model's predictions
            prediction = model.predict(X_test).flatten()

            # TODO print to file:
            # TODO first line: prediction
            # TODO second line:  according y_test - labels
            print(out_file)
            with open(out_file, 'w+') as f:
                f.write(",".join([str(x) for x in prediction]) + "\n")
                f.write(",".join([str(x) for x in y_test]) + "\n")
