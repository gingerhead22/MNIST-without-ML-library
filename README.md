# MNIST-without-ML-library

The repository made a 3-layers neural network without any machine learning library, working on classic MNIST problem.

The dataset can be downloaded from:
http://yann.lecun.com/exdb/mnist/

Please put the code in the same place with a folder named "data", and put all four MNIST files into data folder after extraction.

Currently the model can generate a 93% accuracy on test data

In the accuracy plot, the training accuracy is lower because dropout(r = 0.4) was used during training but not validation
