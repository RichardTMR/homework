import mxnet as mx
import numpy as np
from mxnet import nd


# ctx = mx.cpu()

def mlp_layer(input_layer, n_hidden, activation=None, BN=False):
    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():
    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer, if_pool=False):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?
    W1 = nd.random_normal(shape=(20, 1, 3, 3)) * .01
    b1 = nd.random_normal(shape=20) * .01

    W2 = nd.random_normal(shape=(50, 20, 5, 5)) * .01
    b2 = nd.random_normal(shape=50) * .01

    W3 = nd.random_normal(shape=(800, 128)) * .01
    b3 = nd.random_normal(shape=128) * .01

    W4 = nd.random_normal(shape=(128, 10)) * .01
    b4 = nd.random_normal(shape=10) * .01

    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    conv = nd.Convolution(data=input_layer, weight=W1, bias=b1, kernel=(3, 3), num_filter=20)
    pool = nd.Pooling(data=conv, pool_type="max", kernel=(2, 2), stride=(2, 2))
    print(pool.shape)
    print(conv.shape)
    return pool


def relu(X):
    return nd.maximun(X, nd.zeros_like(X))


def softmax(y_linear):
    exp = nd.exp(y_linear - nd.max(y_linear))
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition


# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    pass


def get_conv_sym(debug=True):
    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")

    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?
    l = conv_layer(input_layer=data)
    l = conv_layer(input_layer=l)
    l = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return l
