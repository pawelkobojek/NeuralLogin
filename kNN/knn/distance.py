import numpy as np
from scipy import spatial

class EuclideanDistance(object):
    def name(self):
        return "euclidean"

    def compute(self, x, y):
        return np.linalg.norm(x - y)

class MaxDistance(object):
    def name(self):
        return "max"

    def compute(self, x, y):
        return np.max(abs(x - y))

class ManhattanDistance(object):
    def name(self):
        return "mahattan"

    def compute(self, x, y):
        return spatial.distance.cityblock(x, y)

class BraycurtisDistance(object):
    def name(self):
        return "braycurtis"

    def compute(self, x, y):
        return spatial.distance.braycurtis(x, y)

class CanberraDistance(object):
    def name(self):
        return "canberra"

    def compute(self, x, y):
        return spatial.distance.canberra(x, y)

class ChebyshevDistance(object):
    def name(self):
        return "chebyshev"

    def compute(self, x, y):
        return spatial.distance.chebyshev(x, y)

class CorrelationDistance(object):
    def name(self):
        return "correlation"

    def compute(self, x, y):
        return spatial.distance.correlation(x, y)

class CosineDistance(object):
    def name(self):
        return "cosine"

    def compute(self, x, y):
        return spatial.distance.cosine(x, y)

class DiceDistance(object):
    def name(self):
        return "dice"

    def compute(self, x, y):
        return spatial.distance.dice(x, y)

class MatchingDistance(object):
    def name(self):
        return "matching"

    def compute(self, x, y):
        return spatial.distance.matching(x, y)

class RussellraoDistance(object):
    def name(self):
        return "russellrao"

    def compute(self, x, y):
        return spatial.distance.russellrao(x, y)

class SokalMichenerDistance(object):
    def name(self):
        return "sokalmichener"

    def compute(self, x, y):
        return spatial.distance.sokalmichener(x, y)

class SokalSneathDistance(object):
    def name(self):
        return "sokalsneath"

    def compute(self, x, y):
        return spatial.distance.sokalsneath(x, y)

class HammingDistance(object):
    def name(self):
        return "hamming"

    def compute(self, x, y):
        return spatial.distance.hamming(x, y)

def all_distances():
    return [EuclideanDistance(), MaxDistance(), ManhattanDistance(),
            BraycurtisDistance(), CanberraDistance(), ChebyshevDistance(),
            CosineDistance(), DiceDistance(), HammingDistance(),
            MatchingDistance(), RussellraoDistance(),
            SokalMichenerDistance(), SokalSneathDistance()]
