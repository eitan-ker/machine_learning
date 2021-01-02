#import math
import math
import random
import sys
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matan_modules
from torchvision import datasets
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


def main():
    train_loader, test_loader = load_data(64)
    model = matan_modules.ModelD(image_size=28*28)
    lr = 0.0011
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(1, 10 + 1):
        train(epoch, model, train_loader, optimizer)
    ans = predict(model, test_loader)
    print_output(ans)



def main2():
    train_loader, test_loader = load_data_and_split_for_test(64)
    model = matan_modules.ModelD(image_size=28*28)
    lr = 0.0011
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(1, 10 + 1):
        train(epoch, model, train_loader, optimizer)
        test(model, test_loader)
    #print_output(ans)

def print_output(y):
    file_output = open("./test_y", 'w')
    for y_hat in y:
        file_output.write(str(y_hat) + "\n")
    file_output.close()


def predict(model, test_loader):
    model.eval()
    arr = []
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred = int(pred)
            arr.append(pred)


    return arr

def check_lr(train_loader, test_loader):
    arr = [0 for i in range(10)]
    for i in range(5):
        lr = 0.00105
        for j in range(10):
            model = matan_modules.ModelE(image_size=28*28)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(1, 10 + 1):
                train(epoch, model, train_loader, optimizer)
            print(str(lr))
            pr = test(model, test_loader)
            arr[j] += pr
            lr += 0.0001

    for i in range(len(arr)):
        arr[i] /= 5
    print(arr)
    print(np.argmax(arr))


def train(epoch, model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels,  reduction='sum')
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target,  reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct/len(test_loader.dataset)


def gettestAndDataset():
    file_trainX = open(sys.argv[1], "r")
    file_trainY = open(sys.argv[2], "r")
    file_testX = open(sys.argv[3], "r")
    try:
        trainX = np.load(sys.argv[1] + '.npy')
        trainY = np.load(sys.argv[2] + '.npy')
        testX = np.load(sys.argv[3] + '.npy')
        # print("after loading data without exeption")
    except OSError:
        trainX = np.loadtxt(file_trainX)
        trainY = np.loadtxt(file_trainY)
        testX = np.loadtxt(file_testX)
    # print("after loading data")
    trainX = np.array(trainX, dtype=np.longdouble)
    trainY = np.array(trainY, dtype=np.int16)
    testX = np.array(testX, dtype=np.longdouble)

    return trainX, trainY, testX


def load_data(batch_size):
    trainX, trainY, testX = gettestAndDataset()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_x = transform(trainX).float().reshape(transform(trainX).size(1), 784)
    trainY = torch.from_numpy(trainY)
    trainY = trainY.type(torch.LongTensor)
    train_ds = TensorDataset(train_x, trainY)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_x = transform(testX).float().reshape(transform(testX).size(1), 784)
    test_ds = TensorDataset(valid_x)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, valid_x


def z_score_normalization(data):
    temp = np.array(data)
    v = temp

    mean = np.mean(v)
    temp_sd = (np.sum(np.square(v)) / len(data)) - mean ** 2
    sd = math.sqrt(temp_sd)

    temp = (v - mean) / sd


    return mean, sd


def splitTrain(trainX, trainY, rate):
    array = []
    for i in range(len(trainX)):
        array.append([trainX[i], trainY[i]])
    random.shuffle(array);
    splitIndex = int(len(trainX) * rate)
    testData = array[splitIndex:]
    array = array[0:splitIndex]
    tempTrainX = []
    tempTrainY = []
    for i in range(len(array)):
        tempTrainX.append(array[i][0])
        tempTrainY.append(array[i][1])
    tempTestData = []
    tempTestAns = []
    for i in range(len(testData)):
        tempTestData.append(testData[i][0])
        tempTestAns.append(testData[i][1])
    tempTrainX = np.array(tempTrainX)
    tempTrainY = np.array(tempTrainY)
    tempTestData = np.array(tempTestData)
    tempTestAns = np.array(tempTestAns)
    return tempTrainX, tempTrainY, tempTestData, tempTestAns


def load_data_from_pytorch(batch_size):
    trainX, trainY, testX = gettestAndDataset()
    trainX = trainX/255
    mean, sd = z_score_normalization(trainX)
    print(mean)
    print(sd)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainX = transform(trainX).float().reshape(transform(trainX).size(1), 784)
    trainY = torch.from_numpy(trainY)
    trainY = trainY.type(torch.LongTensor)
    train_ds = TensorDataset(trainX, trainY)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_ds = datasets.FashionMNIST("../fashion_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_data_and_split_for_test(batch_size):
    trainX, trainY, testX = gettestAndDataset()
    mean, sd = z_score_normalization(trainX)
    print(mean)
    print(sd)
    temp_train_x, temp_train_y, temp_test_data, temp_test_ans = splitTrain(trainX, trainY, 0.8)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (255,))])
    temp_train_x = transform(temp_train_x).float().reshape(transform(temp_train_x).size(1), 784)
    temp_train_y = torch.from_numpy(temp_train_y)
    temp_train_y = temp_train_y.type(torch.LongTensor)
    train_ds = TensorDataset(temp_train_x, temp_train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_x = transform(temp_test_data).float().reshape(transform(temp_test_data).size(1), 784)
    temp_test_ans = torch.from_numpy(temp_test_ans)
    temp_test_ans = temp_test_ans.type(torch.LongTensor)
    test_ds = TensorDataset(valid_x, temp_test_ans)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    main()
    #train_loader, test_loader = load_data_from_pytorch(64)
    #check_lr(train_loader, test_loader)