import sys
import numpy as np
import scipy
import time

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def main():
    y_hat_array = []
    # initialize
    w1 = np.random.uniform(-1, 1, (128, 784))
    b1 = np.random.uniform(-1, 1, (128, 1))
    w2 = np.random.uniform(-1, 1, (10, 128))
    b2 = np.random.uniform(-1, 1, (10, 1))
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # read all data
    train_x, train_y, test_x = loadData()
    # normalize data
    train_x, test_x = normalizeData(train_x, test_x)

    checkAlgo(train_x, train_y, params)

    # epochs = 40
    # lr = 0.0128
    # params = trainAlgo(train_x, train_y, params, epochs, lr)
    #
    # y_hat_array = predictation(test_x, params)
    #
    # file = open("test_y", "w")
    # for i in y_hat_array:
    #     file.write(str(i)+"\n")


def predictation(test_x, params):
    y_hat_array = []
    for x in test_x:
        x = np.reshape(x, (1, len(x)))
        fprop_cache = fprop(x, 0, params)
        y_hat = np.argmax(fprop_cache['h2'], axis = None, out = None)
        y_hat_array.append(y_hat)
    return y_hat_array

def trainAlgo(train_x, train_y, params, epochs, lr):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    for i in range(epochs):
        # print()
        # print("*************")
        # print(f"epoch: {i+1}")
        # print("*************")
        # print()
        zip_arr = list(zip(train_x, train_y))
        np.random.shuffle(zip_arr)
        train_x, train_y = zip(*zip_arr)
        for x, y in zip(train_x, train_y):
            x = np.reshape(x,(1,len(x)))
            # fprop
            # calculate loss
            fprop_cache = fprop(x, y, params)
            # bprop
            bprop_cache = bprop(fprop_cache)
            # update param
            w1 = w1 - (lr * bprop_cache['dw1'])
            w2 = w2 - (lr * bprop_cache['dw2'])
            b1 = b1 - (lr * bprop_cache['b1'])
            b2 = b2 - (lr * bprop_cache['b2'])
            params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return params

def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    y_hat = h2
    y_hat[y] = y_hat[y]-1
    dz2 = y_hat
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['w2'].T,
                 (dz2)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'dw1': dW1, 'b2': db2, 'dw2': dW2}


def fprop(x, y, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    x = np.transpose(x)
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    loss = 0                #change loss function
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret

def softmax(param):
    param = param - np.max(param)
    expParam = np.exp(param)
    ret = expParam / expParam.sum( keepdims=True)
    return ret


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

def checkAlgo(train_x, train_y, params):
    all_arrays = []

    for i in range(5):
        print()
        print("**********")
        print(f"iter: {i}")
        print("**********")
        print()
        temp_train_x_arr, temp_test_x_arr = splitData(train_x, i)
        temp_train_y_arr, temp_test_y_arr = splitData(train_y, i)

        temp_train_x = np.array(temp_train_x_arr)
        temp_test_x = np.array(temp_test_x_arr)
        temp_train_y = np.array(temp_train_y_arr)
        temp_test_y = np.array(temp_test_y_arr)

        accuracy_array = testEpochEta(temp_train_x, temp_train_y, temp_test_x, temp_test_y, params)
        all_arrays.append(accuracy_array)

    avg = avgargedArray(all_arrays)

    best_epoch, best_eta = findOptimal(avg)
    print()
    print("***************************")
    print()
    print(f"avg: {avg}, best_eta: {best_eta}")
    print()
    print("***************************")
    print()


def splitData(data, index):
    train = []
    test = []
    data_size = len(data)
    for i in range(data_size):
        if (i >= (index*(data_size/5)) and i <= ((index+1)*(data_size/5))):
            test.append(data[i])
        else :
            train.append((data[i]))
    return train, test

def testEpochEta(temp_train_x, temp_train_y, temp_test_x, temp_test_y, params):
    accuracy_array = []
    epoch_to_check = 25
    for epoch in range(1):
        epoch_to_check = (epoch_to_check * 2)
        eta_to_check = 0.005
        accuracy_array.append([])
        for eta in range(1):
            start_time = time.time()

            eta_to_check = eta_to_check + 0.0005
            fresh_params = trainAlgo(temp_train_x, temp_train_y, params, 40, eta_to_check)
            yhat = predictation(temp_test_x, fresh_params)
            avg = checkAccuracy(yhat, temp_test_y)
            accuracy_array[epoch].append([avg, eta_to_check])

            print()
            print("**************************************************************")
            print(f"epoch: {40}, eta: {0.007}, avg: {avg}, minutes: {(time.time() - start_time)/60}")
            print("**************************************************************")
            print()

    return accuracy_array

def checkAccuracy(KNN_yhat, temp_test_y):
    counter = 0
    for i in range(len(KNN_yhat)):
        if KNN_yhat[i] == temp_test_y[i]:
            counter += 1
    avg = counter / len(KNN_yhat)
    return avg

def avgargedArray(arrayOfAvarages):
    avg_array = []
    sum = 0
    for epoch in range(len(arrayOfAvarages[0])):
        avg_array.append([])
        for eta in range(len(arrayOfAvarages[0][0])):
            for i in range(5):
                sum = sum + arrayOfAvarages[i][epoch][eta][0]
            avg_array[epoch].append([sum/5, arrayOfAvarages[i][epoch][eta][1]])
            sum = 0
    return avg_array

def findOptimal(algo_avg):

    best_eta = 0
    best_epoch = 0
    avg = 0

    for epoch in range(len(algo_avg)):
        max_avg = max(algo_avg[epoch])
        if avg < max_avg[0]:
            avg = max_avg[0]
            best_epoch = epoch
            best_eta = max_avg[1]
    print(avg)
    return best_epoch, best_eta


if __name__ == "__main__":
    main()