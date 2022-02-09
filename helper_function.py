import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
def initialize_parameters_he(layers_dims):
    param = {}
    L= len(layers_dims) - 1
    for l in range(1, L + 1):
        param['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2./layers_dims[l - 1])
        param['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return param


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, param):
    caches = []
    A = X
    L = len(param) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, param['W' + str(l)], param['b' + str(l)], activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, param['W' + str(L)], param['b' + str(L)], activation='sigmoid')
    caches.append(cache)
    return AL, caches
def cost(AL, Y):
    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
    cost = np.squeeze(cost)
    return cost
def cost_with_L2(AL, Y, param, lambd):
    m = Y.shape[1]
    L = len(param) // 2
    L2_without_cof = 0
    cross_entropy_cost = cost(AL, Y)
    for l in range(1, L):
        L2_without_cof += np.sum(np.square(param['W' + str(l)]))
    L2 = (lambd / 2) * L2_without_cof
    cost_L2 = cross_entropy_cost + L2
    return cost_L2

def linear_backward(dZ, cache, lambd, L2):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    if L2 == True:
        dW = 1./m * np.dot(dZ, A_prev.T) + (lambd / m) * W
    elif L2 == False:
        dW = np.dot(dZ, A_prev.T) / m
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
def sigmoid_backward(dA, cache):
    Z = cache
    S = 1 / (1 + np.exp(-Z))
    dZ = dA * S * (1 - S)
    return dZ

def linear_activation_backward(dA, cache, activation,  lambd, L2):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd, L2=L2)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd, L2=L2)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, lambd, L2):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid",  lambd=lambd, L2=L2)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, activation="relu",  lambd=lambd, L2=L2)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parametrs_with_gd(param, grads, lr):
    L = len(param) // 2
    for l in range(L):
        param['W' + str(l + 1)] -= lr * grads['dW' + str(l + 1)]
        param['b' + str(l + 1)] -= lr * grads['db' + str(l + 1)]
    return param

def initialize_adam(param):
    L = len(param) // 2
    v = {}
    s = {}
    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros((param['W' + str(l + 1)].shape))
        v['db' + str(l + 1)] = np.zeros((param['b' + str(l + 1)].shape))
        s['dW' + str(l + 1)] = np.zeros((param['W' + str(l + 1)].shape))
        s['db' + str(l + 1)] = np.zeros((param['b' + str(l + 1)].shape))
    return v, s

def update_parametrs_with_adam(param, grads, v, s, t = 2, lr = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    L = len(param) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - (beta1) ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - (beta1) ** t)

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * (grads['dW' + str(l + 1)] ** 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * (grads['db' + str(l + 1)] ** 2)

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - (beta2) ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - (beta2) ** t)

        param["W" + str(l + 1)] -= lr * v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        param["b" + str(l + 1)] -= lr * v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return param, v, s

def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_minibatches * mini_batch_size :]
            mini_batch_Y = shuffled_Y[:, num_minibatches * mini_batch_size :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    return mini_batches

def predict(X, Y, param):
    m = X.shape[1]
    n = len(param) // 2
    p = np.zeros((1, m))
    AL, caches = L_model_forward(X, param)
    p[AL > 0.5] = 1
    p[AL <= 0.5] = 0
    print('Accuracy: ', str(np.sum((p == Y) / m)))
    return p

def error_analysis(Y, p, image_names):
    misrecognized = []
    Y = np.squeeze(Y)
    p = np.squeeze(p)
    for i in range(len(Y)):
        if Y[i] != p[i]:
            misrecognized.append(image_names[i])
    fig = plt.figure(figsize=(16, 16))
    columns = 10
    rows = 10
    for i in range(1, columns * rows + 1):
        img = Image.open('/Users/dimasyrovitsky/Documents/nn_without_frameworks/dataset/' + misrecognized[i])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()