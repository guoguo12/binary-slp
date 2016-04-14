#!/usr/bin/env python

import pickle
import slp

CLASSES       = [str(x) for x in xrange(10)]
TRAINING_FILE = '/home/guogu_000/data/mnist_train.csv'
TEST_FILE     = '/home/guogu_000/data/mnist_test.csv'
HOLDOUT       = 3000
TRAIN_ITERS   = 100
FINAL_MODEL   = 'final.mdl'


def readData(path):
    labels = []
    features = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            rawData = map(int, line.strip().split(','))
            labels.append(str(rawData[0]))
            features.append(rawData[1:])
    return features, labels


def validate(slp, validateFeatures, validateLabels):
    right = 0.0
    for feature, label in zip(validateFeatures, validateLabels):
        if label == slp.predict(feature):
            right += 1
    return right / len(validateFeatures)


def main():
    print 'Loading data files...'

    allTrainingFeatures, allTrainingLabels = readData(TRAINING_FILE)
    testFeatures, testLabels = readData(TEST_FILE)

    holdoutIndex = len(allTrainingFeatures)-HOLDOUT
    trainingFeatures, trainingLabels = \
        allTrainingFeatures[:holdoutIndex], allTrainingLabels[:holdoutIndex]
    validateFeatures, validateLabels = \
        allTrainingFeatures[holdoutIndex:], allTrainingLabels[holdoutIndex:]

    featureDim = len(trainingFeatures[0])
    classifier = slp.MulticlassSingleLayerBinaryPerceptron(CLASSES, featureDim)

    print 'Training features: %d' % len(trainingFeatures)
    print 'Validation features: %d' % len(validateFeatures)
    print 'Feature size: %d' % featureDim

    for i in xrange(TRAIN_ITERS):
        print 'Starting training iteration %d' % (i + 1)
        classifier.train(trainingFeatures, trainingLabels)

        accuracy = validate(classifier, validateFeatures, validateLabels)
        print 'Validation accuracy: %f%%' % (accuracy * 100)

    accuracy = validate(classifier, testFeatures, testLabels)
    print 'Test accuracy: %f%%' % (accuracy * 100)

    print 'Writing final model to %s' % FINAL_MODEL
    with open(FINAL_MODEL, 'w') as file:
        pickle.dump(classifier, file)

if __name__ == '__main__':
    main()
