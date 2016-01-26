from __future__ import print_function
import sys, os
sys.path.append("../../")
import dataset
import gather_results

import numpy as np
import theano
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

from emails import get_emails

class Config(object):
    dropout             = 0.5
    batch_size          = 32
    max_seq_len         = 100
    base_save_model_dir = "models"
    activation          = 'sigmoid'
    optimizer           = 'adam'
    loss_function       = 'binary_crossentropy'
    class_mode          = "binary"
    epochs              = 100

    def build_model(self, max_value):
        raise NotImplementedError("Abstract method!")

    def additional_data_transform(self, X_train, X_test):
        raise NotImplementedError("Abstract method!")

class LSTM2Layers2DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "lstm2layer2dropout")
    results_file    = os.path.join(save_model_dir, "total_results.txt")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer = 250, input_length = 100):
        self.first_layer = first_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_model(self, max_value = -1):
        model = Sequential()
        model.add(LSTM(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.input_length))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        return model

    def additional_data_transform(self, X_train, X_test):
        X_train = np.reshape(X_train, (X_train.shape[0], config.max_seq_len, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], config.max_seq_len, 1))
        return X_train, X_test


class LSTM2Layers1DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "lstm2layer1dropout")
    results_file    = os.path.join(save_model_dir, "total_results.txt")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer = 250, input_length = 100):
        self.first_layer = first_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_model(self, max_value = -1):
        model = Sequential()
        model.add(LSTM(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.add(LSTM(self.input_length))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        return model

    def additional_data_transform(self, X_train, X_test):
        X_train = np.reshape(X_train, (X_train.shape[0], config.max_seq_len, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], config.max_seq_len, 1))
        return X_train, X_test

class Embed2LSTMConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "embed2lstm")
    results_file    = os.path.join(save_model_dir, "total_results.txt")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, embed_space_size = 128):
        self.embed_space_size = embed_space_size
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_model(self, max_value):
        model = Sequential()
        model.add(Embedding(max_value, 1))
        model.add(LSTM(self.embed_space_size))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        return model

    def additional_data_transform(self, X_train, X_test):
        # Do nothing
        return X_train, X_test

def train_on_full_data(config, mail, X_train, y_train):
    print("Training on full data")
    X_train = sequence.pad_sequences(X_train, maxlen=config.max_seq_len)
    X_train, X_train = config.additional_data_transform(X_train, X_train)
    max_value = X_train.max() + 1
    print('Build model...')
    model = config.build_model(max_value)
    model.compile(loss=config.loss_function,
                  optimizer=config.optimizer,
                  class_mode=config.class_mode)
    print("Train...")
    model.fit(X_train, y_train, batch_size=config.batch_size, nb_epoch=config.epochs)
    model_path         = os.path.join(config.save_model_dir, mail, "full")
    model_object_path  = os.path.join(model_path, "model.json")
    model_weights_path = os.path.join(model_path, "weights.h5")
    save_model(model, model_object_path, model_weights_path)

def save_model(model, model_file, model_weights_file):
    if not os.path.exists(os.path.dirname(model_file)):
        os.makedirs(os.path.dirname(model_file))
    if not os.path.exists(os.path.dirname(model_weights_file)):
        os.makedirs(os.path.dirname(model_weights_file))
    open(model_file, 'w+').write(model.to_json())
    model.save_weights(model_weights_file, overwrite=True)

def load_model(model_file, model_weights_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(model_weights_file)
    return model

def appendResults(result_string, results_file):
    if not os.path.exists(os.path.dirname(results_file)):
        os.makedirs(os.path.dirname(results_file))
    open(results_file, 'a+').write(result_string + "\n")

def load_and_run_model(config, model_file, model_weights_file, threshold, X_data, y_data):
    model = load_model(model_file, model_weights_file)

    X_data = sequence.pad_sequences(X_data, maxlen=config.max_seq_len)
    X_data, X_data = config.additional_data_transform(X_data, X_data)

    pred = model.predict(X_data)
    return np.mean(np.abs(y_data - (pred.T > threshold)))

def get_threshold(thresholds_dir, mail):
    with open(os.path.join(thresholds_dir, mail + ".txt")) as f:
        return float(f.read())

def load_and_run_all_models(config, emails_list_file, emails_data_base_dir):
    emails = get_emails(emails_list_file)
    total_err = 0.0
    for mail in emails:
        X_data, Y_data     = dataset.load_data(mail, emails_data_base_dir)
        model_path         = os.path.join(config.save_model_dir, mail, "full")
        model_object_path  = os.path.join(model_path, "model.json")
        model_weights_path = os.path.join(model_path, "weights.h5")
        threshold          = get_threshold(config.thresholds_dir, mail)

        err = load_and_run_model(config, model_object_path, model_weights_path,
                                            threshold, X_data, Y_data)
        print("Error for %s with threshold %f: %f" % (mail, threshold, err))
        total_err = (total_err + err) / 2

    print("Total error:", total_err)

def get_result(mail, results_dir):
    filename = os.path.join(results_dir, mail + ".txt")
    with open(filename) as f:
        resultset = [d.split(',') for d in f.read().splitlines()]
        y_score = resultset[0]
        y = resultset[1]
        y_score = [float(x) for x in y_score]
        y = [int(x) for x in y]
        return y_score, y

def false_positives(y_score, y, threshold):
    fp = 0
    for i in range(len(y)):
        p = int(y_score[i] > threshold)
        if p == 1 and y[i] == 0:
            fp += 1

    return fp

def find_threshold(y_score, y):
    best_th = 0
    best_false_positives = len(y)
    for t in np.arange(0.50, 0.9999, 0.0001):
        fp = false_positives(y_score, y, t)
        if fp < best_false_positives:
            best_th = t
            best_false_positives = fp

    return best_th

def compute_thresholds(out_results_dir, thresholds_dir, emails_list_file):
    if not os.path.exists(thresholds_dir):
        os.makedirs(thresholds_dir)

    emails = get_emails(emails_list_file)
    for mail in emails:
        y_score, y = get_result(mail, out_results_dir)

        th = find_threshold(y_score, y)
        th_file = os.path.join(thresholds_dir, mail + ".txt")
        print("Saving threshold:", th_file)
        with open(th_file, "w+") as f:
            f.write(str(th))


def train_and_evaluate(config, emails_list_file, emails_data_base_dir):
    emails = get_emails(emails_list_file)
    totalFP = 0
    totalFN = 0
    totalTP = 0
    totalTN = 0
    with open(config.results_file, "w+") as f:
        f.write("")
    for mail in emails:

        print('Loading data...')
        X_data, Y_data = dataset.load_data(mail, emails_data_base_dir)

        falsePositives = 0
        falseNegatives = 0
        truePositives  = 0
        trueNegatives  = 0
        predictions = []
        shouldBe = []
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

            print(len(X_train), 'train sequences')

            print("Pad sequences (samples x time)")
            X_train = sequence.pad_sequences(X_train, maxlen=config.max_seq_len)
            X_test = sequence.pad_sequences(X_test, maxlen=config.max_seq_len)
            X_train, X_test = config.additional_data_transform(X_train, X_test)
            max_value = max(X_train.max(), X_test.max()) + 1
            print('X_train shape:', X_train.shape)
            print('X_test shape:', X_test.shape)

            print('Build model...')
            model = config.build_model(max_value)
            model.compile(loss=config.loss_function,
                          optimizer=config.optimizer,
                          class_mode=config.class_mode)

            print("Train...")
            model.fit(X_train, y_train, batch_size=config.batch_size, nb_epoch=config.epochs,
                      validation_data=(X_test, y_test), show_accuracy=True)
            score, acc = model.evaluate(X_test, y_test,
                                        batch_size=config.batch_size,
                                        show_accuracy=True)

            prediction = model.predict(X_test)
            predicted_class = round(abs(float(prediction)))
            print("Model prediction:", prediction, "Should be:", y_test)
            print('Test score:', score)
            print('Test accuracy:', acc)

            predictions.append(float(prediction))
            shouldBe.append(y_test[0])

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

            model_path         = os.path.join(config.save_model_dir, mail, str(i))
            model_object_path  = os.path.join(model_path, "model.json")
            model_weights_path = os.path.join(model_path, "weights.h5")

            save_model(model, model_object_path, model_weights_path)

        train_on_full_data(config, mail, X_data, Y_data)

        result_string = "\n".join(["For: %s FP: %d FN: %d TP: %d TN: %d",
                                    "Predictions: %s", "Should be: %s"])
        result_string = result_string % (mail, falsePositives, falseNegatives,
                        truePositives, trueNegatives, str(predictions),
                        str(shouldBe))
        appendResults(result_string, config.results_file)
        print(result_string)


    result_string = "TotalFP: " + str(totalFP) + " TotalFN: " + str(totalFN) + \
                    " TotalTP: " + str(totalTP) + " TotalTN: " + str(totalTN)
    appendResults(result_string, config.results_file)
    print(result_string)
    gather_results.from_file(config.results_file, config.out_results_dir)
    print("Looking for FP-free threshold...")
    compute_thresholds(config.out_results_dir, config.thresholds_dir, emails_list_file)


if __name__ == "__main__":
    emails_list_file     = "../../emails.txt"
    emails_data_base_dir = "../../training_data"
    emails_test_data_dir = "../../test_data"
    if len(sys.argv) > 1:
        emails_list_file = argv[1]
    if len(sys.argv) > 2:
        emails_data_base_dir = argv[2]
    if len(sys.argv) > 3:
        emails_test_data_dir = argv[3]

    configs = [
            Embed2LSTMConfig(),
            LSTM2Layers1DropoutsConfig(),
            LSTM2Layers2DropoutsConfig()
        ]
    for config in configs:
        train_and_evaluate(config, emails_list_file, emails_data_base_dir)
        load_and_run_all_models(config, emails_list_file, emails_data_base_dir)
