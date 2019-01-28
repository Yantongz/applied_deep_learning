import numpy as np
import random
import h5py

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def softmax(z):
    aa = np.exp(z)/np.sum(np.exp(z), axis = 0)
    return aa


def init_params(X, num_hid_units, num_labels):
    # initialize weights matrix (shrink the variance of the weights in each layer) and bias vectors
    params = {}
    params['W1'] = np.random.randn(num_hid_units, X.shape[0]) * np.sqrt(1. / X.shape[0])
    params['b1'] = np.zeros((num_hid_units, 1))
    params['W2'] = np.random.randn(num_labels, num_hid_units) * np.sqrt(1. / num_hid_units)
    params['b2'] = np.zeros((num_labels, 1))
    return params


def update_params(params, gradients, lr):

    # update weights and bias with learning rate.
    # lr : the learning rate
    params['W1'] -= lr * gradients['dJ_dW1']
    params['b1'] -= lr * gradients['dJ_db1']
    params['W2'] -= lr * gradients['dJ_dW2']
    params['b2'] -= lr * gradients['dJ_db2']
    return params


def loss_func(A2,Y):
    # calculate the error between predicted class and actual class

    m = Y.shape[1]
    J = -1.0/m * np.sum(Y * np.log(A2))
    return J


def forward_prop(X, Y, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2) # softmax for multi-class

    layer = {}
    layer['A1'] = A1
    layer['A2'] = A2
    layer['Z1'] = Z1
    layer['Z2'] = Z2

    return layer


def backward_prop(X, Y, layer, params):

    m = Y.shape[1] # sample size
    W2 = params['W2']
    A1 = layer['A1']
    A2 = layer['A2']
    Z1 = layer['Z1']

    dJ_dZ2 = A2 - Y
    dJ_dW2 = 1. / m * np.dot(dJ_dZ2, A1.T)
    dJ_db2 = 1. / m * np.sum(dJ_dZ2, axis=1, keepdims=True)

    dJ_dA1 = np.dot(W2.T, dJ_dZ2)
    dJ_dZ1 = dJ_dA1 * sigmoid(Z1)*(1-sigmoid(Z1))
    dJ_dW1 = 1. / m * np.dot(dJ_dZ1, X.T)
    dJ_db1 = 1. / m * np.sum(dJ_dZ1, axis=1, keepdims=True)

    gradients = {}
    gradients['dJ_dW1'] = dJ_dW1
    gradients['dJ_dW2'] = dJ_dW2
    gradients['dJ_db1'] = dJ_db1
    gradients['dJ_db2'] = dJ_db2
    return gradients


# train the model
# iterations : number of batches to complete one epoch
# batch_size : the batch size, default = 50

def train_model(X, Y, num_labels, num_hid_units, iterations, lr, batch_size = 50):

    i_param = init_params(X, num_hid_units, num_labels)
    sample_size = Y.shape[1]

    for i in range(iterations):
        index = np.arange(sample_size)
        random.shuffle(index)

        for j in range(sample_size // batch_size):

            start = j * batch_size
            end = min((j + 1) * batch_size, sample_size)
            X_batch = X[:, index[start:end]]
            Y_batch = Y[:, index[start:end]]

            layer_j = forward_prop(X_batch, Y_batch, i_param)
            gradients = backward_prop(X_batch, Y_batch, layer_j, i_param)
            i_param = update_params(i_param, gradients, lr)

    return i_param


# make prediction based on the trained parameters
def prediction(X, y, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    y_hat = A2.argmax(axis=0)
    accuracy = sum(y_hat == y) / len(y)
    return y_hat, accuracy


if __name__ == '__main__':
    # load MNIST data
    MNIST_data = h5py.File('C:/Users/yantongz/Desktop/IE534/MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
    MNIST_data.close()

    # organize the data
    X_train = x_train.T
    X_test = x_test.T
    num_labels = len(set(y_train))

    # one hot encoding
    Y_train = np.zeros((num_labels, len(y_train)))
    for i in range(len(y_train)):
        Y_train[y_train[i]][i] = 1

    iterations = 100
    batch_size = 50

    train_accuracy = {}
    test_accuracy = {}
    # try different kinds of learning rate and hidden units
    for poss_lr in [ 0.1, 0.5, 1, 2]:
        train_accuracy[poss_lr] = {}
        test_accuracy[poss_lr] = {}
        for num_hid_units in [32, 64]:
            # train the model and get the parameters
            parameters = train_model(X_train, Y_train, num_labels, num_hid_units, iterations, poss_lr,batch_size)
            # make prediction on the training data set
            train_y_hat, train_accuracy[poss_lr][num_hid_units] = prediction(X_train, y_train, parameters)
            # make prediction on the test data set
            test_y_hat, test_accuracy[poss_lr][num_hid_units] = prediction(X_test, y_test, parameters)

            print("When learning rate is " + str(poss_lr) + " and number of units in the hidden layer is " + str(num_hid_units) + ". ")
            print("Accuracy for training data set: %f" % (train_accuracy[poss_lr][num_hid_units]))
            print("Accuracy for test data set: %f\n" % (test_accuracy[poss_lr][num_hid_units]))