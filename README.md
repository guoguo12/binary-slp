# binary-slp

A binary single-layer perceptron, applied to the [MNIST handwriting recognition task](http://yann.lecun.com/exdb/mnist/).

## Results

This system produces test error rates of 10-15%. According to the MNIST website, this is the error rate expected of a linear classifier when no preprocessing is performed.

Running more iterations of training does not seem to reliably improve accuracy.

## Reproducing results

This program requires a recent version of Python 2 and [NumPy](http://www.numpy.org/).

Download the MNIST data files in CSV format from [this page](http://pjreddie.com/projects/mnist-in-csv/), then set the `TRAINING_FILE` and `TEST_FILE` variables at the top of `main.py` to point to these two files.

To begin training, run `python main.py`. With the default settings, you should see output like this:

```
Loading data files...
Training features: 57000
Validation features: 3000
Feature size: 784
Starting training iteration 1
Validation accuracy: 87.833333%
Starting training iteration 2
Validation accuracy: 85.700000%
Starting training iteration 3
Validation accuracy: 87.366667%
Starting training iteration 4
Validation accuracy: 89.333333%
...
Starting training iteration 99
Validation accuracy: 88.333333%
Starting training iteration 100
Validation accuracy: 87.433333%
Test accuracy: 85.660000%
Writing final model to final.mdl
```
