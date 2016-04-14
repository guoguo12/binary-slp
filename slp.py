#!/usr/bin/env python

import numpy as np


def sign(x):
    return 1 if x >= 0 else -1


class MulticlassSingleLayerBinaryPerceptron(object):

    def __init__(self, classes, featureDim):
        self.classes = classes
        self.weights = [np.zeros(featureDim) for _ in classes]

    def train(self, features, labels):
        for feature, label in zip(features, labels):
            featureArr = np.array(feature)
            for i, weight in enumerate(self.weights):
                product = sign(np.dot(featureArr, weight))
                correctProduct = 1 if self.classes[i] == label else -1
                if product != correctProduct:
                    self.weights[i] += correctProduct * featureArr

    def predict(self, feature):
        featureArr = np.array(feature)
        return max([
            (np.dot(featureArr, weight), label)
            for weight, label in zip(self.weights, self.classes)
        ], key=lambda p: p[0])[1]
