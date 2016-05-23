import os
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2

class Config(object):
    dropout             = 0.5
    batch_size          = 64
    max_seq_len         = 100
    base_save_model_dir = "benchmark_full_results"
    activation          = 'sigmoid'
    optimizer           = 'adam'
    loss_function       = 'binary_crossentropy'
    class_mode          = "binary"
    epochs              = 100

class LSTM3Layers3DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "lstm3layer3dropout")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer, second_layer, input_length):

        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_and_compile(self):
        model = Sequential()
        model.add(LSTM(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.second_layer, return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.second_layer))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer, metrics=["accuracy"])
        return model

class GRU3Layers3DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "gru3layer3dropout")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer, second_layer, input_length):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_and_compile(self):
        model = Sequential()
        model.add(GRU(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(GRU(self.second_layer, return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(GRU(self.second_layer))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer, metrics=["accuracy"])
        return model

class ArtificialLSTM3Layers3DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "art_lstm3layer3dropout")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer, second_layer, input_length):

        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_and_compile(self):
        model = Sequential()
        model.add(LSTM(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.second_layer, return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.second_layer))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer, metrics=["accuracy"])
        return model

class ArtificialGRU3Layers3DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "art_gru3layer3dropout")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer, second_layer, input_length):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_and_compile(self):
        model = Sequential()
        model.add(GRU(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(GRU(self.second_layer, return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(GRU(self.second_layer))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer, metrics=["accuracy"])
        return model

class ArtificialLSTM2Layers2DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "art_lstm2layer2dropout")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer, second_layer, input_length):

        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_and_compile(self):
        model = Sequential()
        model.add(LSTM(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.second_layer))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer, metrics=["accuracy"])
        return model


class OneClassGRU3Layers3DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "oneclassgru3layer3dropout")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer, second_layer, input_length):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_and_compile(self):
        model = Sequential()
        model.add(GRU(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(GRU(self.second_layer, return_sequences=True))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(GRU(self.second_layer))
        model.regularizers.append(l2(0.01))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer, metrics=["accuracy"])
        return model


class LSTM2Layers2DropoutsConfig(Config):
    save_model_dir  = os.path.join(Config.base_save_model_dir, "lstm2layer2dropout")
    out_results_dir = os.path.join(save_model_dir, "results")
    thresholds_dir  = os.path.join(save_model_dir, "thresholds")

    def __init__(self, first_layer, second_layer, input_length):

        self.first_layer = first_layer
        self.second_layer = second_layer
        self.input_length = input_length
        if not os.path.exists(self.out_results_dir):
            os.makedirs(self.out_results_dir)
        if not os.path.exists(self.thresholds_dir):
            os.makedirs(self.thresholds_dir)

    def build_and_compile(self):
        model = Sequential()
        model.add(LSTM(self.first_layer, input_shape=(self.input_length, 1), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.second_layer))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.add(Activation(self.activation))
        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer, metrics=["accuracy"])
        return model
