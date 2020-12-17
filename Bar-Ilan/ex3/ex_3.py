import sys
import numpy as np
import scipy

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def main():
    # initialize
    w1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # read all data
    train_x, train_y, test_x = loadData()
    # normalize data
    train_x, test_x = normalizeData(train_x, test_x)

    epoch = 1
    # loop epoch number of times
    for i in range(epoch):
        zip_arr = list(zip(train_x, train_y))
        np.random.shuffle(zip_arr)
        train_x, train_y = zip(*zip_arr)
        for x, y in zip(train_x, train_y):
            X = x.reshape(x,(1,784))
            # fprop
            # calculate loss
            fprop_cache = fprop(x, y, params)
            # bprop
            bprop_cache = bprop(fprop_cache)
            z=2
            # update param


def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    # y_hat = h2
    # y_hat = y_hat[y]-1
    # dz2 = y_hat
    dz2 = h2[y] - 1  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['w2'].T,
                 (dz2)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'w1': dW1, 'b2': db2, 'w2': dW2}


def fprop(x, y, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    x = np.transpose(x)
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    loss = 0
    train_x = np.transpose(x)
    #change loss function
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret

def softmax(param):
    param = param - np.max(param)
    expParam = np.exp(param)
    ret = expParam / expParam.sum( keepdims=True)
    return ret


def shuffle_data(train_x, train_y):
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    return train_x[indices], train_y[indices]


def normalizeData(train_x, test_x):
    for i in range(len(train_x)):
        train_x[i] = train_x[i]/255
    for i in range(len(test_x)):
        test_x[i] = test_x[i]/255
        return train_x, test_x


def loadData():
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2]).astype(int)
    test_x = np.loadtxt(sys.argv[3])
    return train_x, train_y, test_x


if __name__ == "__main__":
    main()