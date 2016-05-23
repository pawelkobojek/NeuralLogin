import os
import numpy as np

class GaussianGenerator(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def generate(self, example):
        return (example + np.random.normal(self.mean, self.std, len(example))).tolist()

class Dataset(object):
    def __init__(self, base_dir="../../training_data"):
        self.base_dir = base_dir

    def load(self, mail):
        filename = os.path.join(base_dir, mail + ".txt")
        # filename = "../../training_data/" + mail + ".txt"
        with open(filename) as f:
            dataset = [d.split(',') for d in f.read().splitlines()]
            Y_data = [int(c[0]) for c in dataset]
            X_data = [c[1:] for c in dataset]
            X_data = list(map(lambda x: list(map(lambda y: int(y), x)), X_data))

        return X_data, Y_data

class BenchmarkDataset(Dataset):
    def __init__(self, base_dir):
        super(BenchmarkDataset, self).__init__(base_dir)

    def load(self, mail):
        filename = os.path.join(self.base_dir, mail + ".txt")
        # filename = "../../training_data/" + mail + ".txt"
        with open(filename) as f:
            dataset = [d.split(',') for d in f.read().splitlines()]
            Y_data = [int(c[0]) for c in dataset]
            X_data = [c[1:] for c in dataset]
            X_data = list(map(lambda x: list(map(lambda y: float(y), x)), X_data))

        return X_data, Y_data

class ArtrificialDataset(Dataset):
    def __init__(self, base_dir, generator, artrificial_rows_count):
        super(ArtrificialDataset, self).__init__(base_dir)
        self.generator = generator
        self.artrificial_rows_count = artrificial_rows_count

    def load(self, mail):
        X_data, Y_data, negative_X_test, negative_Y_test = self.load_positive_data(mail, self.base_dir)
        j = 0
        initial_size = len(X_data)
        for i in range(initial_size):
            for j in range(self.artrificial_rows_count):
                X_data.append(self.generator.generate(X_data[i]))
                Y_data.append(0)

        return X_data, Y_data, negative_X_test, negative_Y_test

    def load_positive_data(self, mail, base_dir="../../training_data"):
        filename = os.path.join(base_dir, mail + ".txt")
        # filename = "../../training_data/" + mail + ".txt"
        with open(filename) as f:
            dataset = [d.split(',') for d in f.read().splitlines() if d.split(',')[0] == '1']
            Y_data = [int(c[0]) for c in dataset]
            X_data = [c[1:] for c in dataset]
            X_data = list(map(lambda x: list(map(lambda y: int(y), x)), X_data))

            only_negatives = [d.split(',') for d in f.read().splitlines() if d.split(',')[0] == '0']
            negative_Y_test = [int(c[0]) for c in dataset]
            negative_X_test = [c[1:] for c in dataset]
            negative_X_test = list(map(lambda x: list(map(lambda y: int(y), x)), X_data))

        return X_data, Y_data, negative_X_test, negative_Y_test


def load_data(mail, base_dir="../../training_data"):
    filename = os.path.join(base_dir, mail + ".txt")
    # filename = "../../training_data/" + mail + ".txt"
    with open(filename) as f:
        dataset = [d.split(',') for d in f.read().splitlines()]
        Y_data = [int(c[0]) for c in dataset]
        X_data = [c[1:] for c in dataset]
        X_data = list(map(lambda x: list(map(lambda y: int(y), x)), X_data))

    return X_data, Y_data
