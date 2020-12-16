import sys
import numpy
import scipy.spatial.distance as dist
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def main():
    train_x, train_y, test_x = readAllData()
    train_x, test_x = normilizeData(train_x, test_x)
  #  checkAlgo(train_x, train_y)
    KNN_yhat = KNN(7, train_x, train_y, test_x)
    perceptron_yhat = perceptron(train_x, train_y, test_x, 0.6, 19)
    pa_yhat = pa(train_x, train_y, test_x, 0.6, 19)

    for i in range(len(test_x)):
        print(f"knn: {KNN_yhat[i]}, perceptron: {perceptron_yhat[i]}, pa: {pa_yhat[i]}")

def checkAlgo(train_x, train_y):
    knn_all_arrays = []
    per_all_arrays = []
    pa_all_arrays = []

    for i in range(5):
        temp_train_x_arr, temp_test_x_arr = splitData(train_x, i)
        temp_train_y_arr, temp_test_y_arr = splitData(train_y, i)

        temp_train_x = numpy.array(temp_train_x_arr)
        temp_test_x = numpy.array(temp_test_x_arr)
        temp_train_y = numpy.array(temp_train_y_arr)
        temp_test_y = numpy.array(temp_test_y_arr)

        knn_accuracy_array = testKNN(temp_train_x, temp_train_y, temp_test_x, temp_test_y)
        knn_all_arrays.append(knn_accuracy_array)

        per_accuracy_array = testEpochEta(temp_train_x, temp_train_y, temp_test_x, temp_test_y, 0)
        per_all_arrays.append(per_accuracy_array)

        pa_accuracy_array = testEpochEta(temp_train_x, temp_train_y, temp_test_x, temp_test_y, 1)
        pa_all_arrays.append(pa_accuracy_array)

    knn_avg = avgargedKArray(knn_all_arrays)
    per_avg = avgargedPerPaArray(per_all_arrays)
    pa_avg = avgargedPerPaArray(per_all_arrays)

    best_K, best_epoch_per, best_eta_per, best_epoch_pa, best_eta_pa = findOptimal(knn_avg, per_avg, pa_avg)
    print(best_K)
    print(best_epoch_per)
    print(best_eta_per)
    print(best_epoch_pa)
    print(best_eta_pa)


def findOptimal(knn_avg, per_avg, pa_avg):
    best_K = 0
    best_epoch_per = 0
    best_epoch_pa = 0
    best_eta_per = 0
    best_eta_pa = 0
    avg = 0
    for i in range(len(knn_avg)):
        if avg < knn_avg[i]:
            avg = knn_avg[i]
            best_K = i
    avg_per = 0
    avg_pa = 0
    for epoch in range(len(per_avg)):
        max_per = max(per_avg[epoch])
        max_pa = max(pa_avg[epoch])
        if avg_per < max_per[0]:
            avg_per = max_per[0]
            best_epoch_per = epoch
            best_eta_per = max_per[1]
        if avg_pa < max_pa[0]:
            avg_pa = max_pa[0]
            best_epoch_pa = epoch
            best_eta_pa = max_pa[1]
    print(avg)
    print(avg_per)
    print(avg_pa)


    return best_K, best_epoch_per, best_eta_per, best_epoch_pa, best_eta_pa


def avgargedKArray(arrayOfAvarages):
    avg_array = []
    sum = 0
    for j in range(len(arrayOfAvarages[0])):
        for i in range(5):
            sum = sum + arrayOfAvarages[i][j]
        avg_array.append(sum/5)
        sum = 0
    return avg_array

def avgargedPerPaArray(arrayOfAvarages):
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


def testKNN(temp_train_x, temp_train_y, temp_test_x, temp_test_y):
    knn_accuracy_array = []
    for k in range(10):
        KNN_yhat = KNN(k+1, temp_train_x, temp_train_y, temp_test_x)
        avg = checkAccuracy(KNN_yhat, temp_test_y)
        knn_accuracy_array.append(avg)
    return knn_accuracy_array


def testEpochEta(temp_train_x, temp_train_y, temp_test_x, temp_test_y, index):
    accuracy_array = []
    eta_range = 25
    for epoch in range(20):
        accuracy_array.append([])
        for eta in range(eta_range):
            if index == 0:
                yhat = perceptron(temp_train_x, temp_train_y, temp_test_x, eta/eta_range, epoch+1)
            else:
                yhat = pa(temp_train_x, temp_train_y, temp_test_x, eta/eta_range, epoch+1)
            avg = checkAccuracy(yhat, temp_test_y)
            accuracy_array[epoch].append([avg, eta/eta_range])
    return accuracy_array

def checkAccuracy(KNN_yhat, temp_test_y):
    counter = 0
    for i in range(len(KNN_yhat)):
        if KNN_yhat[i] == temp_test_y[i]:
            counter += 1
    avg = counter / len(KNN_yhat)
    return avg



def testForKNN(k, temp_train_x, temp_train_y, temp_test_x, temp_test_y):
    counter_KNN = 0
    KNN_yhat = KNN(k, temp_train_x, temp_train_y, temp_test_x)
    for i in range(len(temp_test_y)):
        if KNN_yhat[i] == temp_test_y[i]:
            counter_KNN += 1
    avg_KNN = counter_KNN / len(temp_test_y)
    return avg_KNN


def splitData(data, index):
    train = []
    test = []
    data_size = len(data)
    for i in range(data_size):
        if (i >= (index*(data_size/5)) and i <= ((index+1)*(data_size/5))):
            test.append(data[i])
        else :
            train.append((data[i]))
    return train,test



def pa(train_x, train_y, test_x, eta, epoches):
    prediction = []
    w = trainWPa(train_x, train_y, eta, epoches)
    for i in range(len(test_x)):
        yhat = labelYHat(w, test_x[i])
        prediction.append(yhat)
    return prediction


def trainWPa(train_x, train_y, eta, epoches):
    w = []
    for i in range(3):
        w.append([])
        for j in range(13):
            if j == 12:
                w[i].append(1)
                break
            else:
                w[i].append(random.rand())
    trainX = train_x.tolist()
    for i in range(len(train_x)):
        trainX[i].append(1)
    num_train_x = numpy.array(trainX)
  #  epoches = 10
    for e in range(epoches):
        zip_arr = list(zip(num_train_x, train_y))
        np.random.shuffle(zip_arr)
        num_train_x, train_y = zip(*zip_arr)
        for x, y in zip(num_train_x, train_y):
            # predict
            tempW = []
            for i in range(len(w)):
                if i != y:
                    tempW.append(w[i])
            y_hat = np.argmax(np.dot(tempW, x))
            if y <= y_hat:
                y_hat += 1
            # update
            if y != y_hat:
                tau = calculateTau(w, x, y, y_hat)
                w[y] = w[y] + eta * x * tau
                w[y_hat] = w[y_hat] - eta * x * tau
    return w


def calculateTau(w, x, y, y_hat):
    temp = 1 - numpy.dot(w[y], x) + numpy.dot(w[y_hat], x)
    l = max(0, temp)
    div = 2 * (numpy.linalg.norm(x) ** 2)
    if div != 0:
        return l / div
    else:
        return 0


def perceptron(train_x, train_y, test_x, eta, epoches):
    prediction = []
    # train - make w correct according to training
    w = trainWPer(train_x, train_y, eta, epoches)
    # use w for test_x
    for i in range(len(test_x)):
        yhat = labelYHat(w, test_x[i])
        prediction.append(yhat)
    return prediction


def labelYHat(w, test_x):
    testX = test_x.tolist()
    testX.append(1)
    num_test_x = numpy.array(testX)
    tempArray = []
    for i in range(len(w)):
        dis = sum(w[i] * num_test_x)
        tempArray.append(dis)
    return tempArray.index(max(tempArray))


def trainWPer(train_x, train_y, eta, epoches):
    w = []
    for i in range(3):
        w.append([])
        for j in range(13):
            if j == 12:
                w[i].append(1)
                break
            else:
                w[i].append(random.rand())
    trainX = train_x.tolist()
    for i in range(len(train_x)):
        trainX[i].append(1)
    num_train_x = numpy.array(trainX)
  #  epoches = 10
    for e in range(epoches):
        zip_arr = list(zip(num_train_x, train_y))
        np.random.shuffle(zip_arr)
        num_train_x, train_y = zip(*zip_arr)
        for x, y in zip(num_train_x, train_y):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            # update
            if y != y_hat:
                w[y] = w[y] + eta * x
                w[y_hat] = w[y_hat] - eta * x
    return w


def KNN(k, trainX, trianY, test_x):
    neighbors = []
    for j in range(len(test_x)):
        neighbors.append([])
        for i in range(len(trainX)):
            dis = dist.euclidean(test_x[j], trainX[i])
            y_label = trianY[i]
            neighbors[j].append([dis, y_label])
    lablelArray = []
    for i in range(len(neighbors)):
        neighborsSubArray = numpy.array(neighbors[i])
        sortedSubArr = neighborsSubArray[neighborsSubArray[:, 0].argsort()]
        classOfKNN = checkClass(sortedSubArr, k)
        lablelArray.append(classOfKNN)
    return lablelArray


def checkClass(sortedArr, k):
    clusters = [0, 0, 0]
    for i in range(k):
        if sortedArr[i][1] == 0:
            clusters[0] = clusters[0] + 1
        elif sortedArr[i][1] == 1:
            clusters[1] = clusters[1] + 1
        else:
            clusters[2] = clusters[2] + 1
    maxCluster = -1
    maxValue = 0
    for i in range(3):
        if maxValue < clusters[i]:
            maxValue = clusters[i]
            maxCluster = i
    return maxCluster


def normilizeData(train_x, test_x):
    numOfFeathers = len(train_x[0])
    # transpose both matrices for easier computation
    train_x = train_x.transpose()
    test_x = test_x.transpose()
    for i in range(numOfFeathers):
        minValue = min(train_x[i])
        maxValue = max(train_x[i])
        if maxValue != minValue:
            train_x[i] = (train_x[i] - minValue) / (maxValue - minValue)
            test_x[i] = (test_x[i] - minValue) / (maxValue - minValue)
    train_x = train_x.transpose()
    test_x = test_x.transpose()
    return train_x, test_x


def readAllData():
    train_x = np.loadtxt(sys.argv[1], delimiter=',', converters={11: lambda s: 1 if s == b'R' else 0})
    train_y = np.loadtxt(sys.argv[2], delimiter=',').astype(int)
    test_x = np.loadtxt(sys.argv[3], delimiter=',', converters={11: lambda s: 1 if s == b'R' else 0})
    return train_x, train_y, test_x


if __name__ == '__main__':
    main()
