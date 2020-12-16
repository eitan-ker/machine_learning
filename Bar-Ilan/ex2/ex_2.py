import sys
import numpy
import math
import random
import itertools
import scipy.spatial.distance as dista
import matplotlib.pyplot as plt

def main():
    trainX, trainY, testX = gettestAndDataset()
    trainX,testX = normilaisData(trainX,testX)
    trainX, testX = addBais(trainX,testX)
    #checkAlgo(trainX,trainY,7,0.03,1)
    Knn_yhat = KNNnerest(trainX,trainY,7,testX);
    Perceptron_yhat = Perceptron(trainX,trainY,testX,3,30,0.03)
    PA_yhat = PAalgo(trainX,trainY,testX,3,34,1)

    for i in range(len(testX)):
        print(f"knn: {Knn_yhat[i]}, perceptron: {Perceptron_yhat[i]}, pa: {PA_yhat[i]}")

def findOptimalLerninigRate(trainX, trainY):
    numOfErrorPerceptron = []
    numOfErrorPa = []
    for j in range(200):
        tempTrainX, tempTrainY, tempTestData, tempTestAns = splitTrain(trainX, trainY, 0.8)
        for i in range (1,40):
            Perceptron_yhat = Perceptron(tempTrainX, tempTrainY, tempTestData, 3,i,1)
            PA_yhat = PAalgo(tempTrainX, tempTrainY, tempTestData, 3, i, 0.34)
            mistakePerceptron = 0
            mistakePa = 0
            for w in range(len(tempTestAns)):
                if Perceptron_yhat[w] != tempTestAns[w]:
                    mistakePerceptron += 1
                if PA_yhat[w] != tempTestAns[w]:
                    mistakePa += 1
            if j==0:
                numOfErrorPerceptron.append(mistakePerceptron / len(tempTestAns))
                numOfErrorPa.append(mistakePa / len(tempTestAns))
            else:
                numOfErrorPerceptron[i-1]+=(mistakePerceptron / len(tempTestAns))
                numOfErrorPa[i-1]+=(mistakePa / len(tempTestAns))
            print("iter "+ str (i)+ "in the iteration" + str(j))
            print("error of Per = " +str((numOfErrorPerceptron[i-1]/(j+1))) + " error of PA = " + str((numOfErrorPa[i-1])/(j+1)))
    plt.plot(range(1, 40), numOfErrorPerceptron, color='green', linestyle='dashed',
                 marker='o', markerfacecolor='red', markersize=10 ,label='Perceptron')
    plt.plot(range(1, 40), numOfErrorPa, color='red', linestyle='dashed',
                 marker='o', markerfacecolor='red', markersize=10 ,label='PA')
    plt.title('Error Rate vs. Lerning Rate Number')
    plt.xlabel('Itaration Number')
    plt.ylabel('Error Rate')
    print(" Perceptron best = " + str(numOfErrorPerceptron[numOfErrorPerceptron.index(min(numOfErrorPerceptron))] /40) + " PA best = " + str(numOfErrorPa[numOfErrorPa.index(min(numOfErrorPa))] /40))
    plt.show()
    return numOfErrorPerceptron.index(min(numOfErrorPerceptron)) *0.01, numOfErrorPa.index(min(numOfErrorPa))*0.01;

def featherSelected(trainX, trainY, testX):
    bestFeather=[];
    bestAccuracy=1
    numOfFeather = len(testX[0])
    array = [i for i in range(numOfFeather)]
    for i in range(int(numOfFeather/2)+3,numOfFeather):
        subset = itertools.combinations(array,i)
        for k in subset:
            newTrainX, newTestX = updateDataWithKelement(k,trainX,testX)
            tempKnn,tempPer,tempPA=checkAlgo(newTrainX, trainY, 7)
            aver= (tempKnn + tempPA + tempPer)/3
            print(aver,k)
            if aver<bestAccuracy:
                bestAccuracy = aver
                bestFeather = k
    print("the best feather is" + str(bestFeather)+"the accuracy is" +str(bestAccuracy))

def updateDataWithKelement(iter,trainX,testX):
    newTrainX=[]
    newTestX =[]
    for i in range(len(trainX)):
        x =[]
        for j in iter:
            x.append(trainX[i][j])
        x = numpy.array(x)
        newTrainX.append(x)
    for i in range(len(testX)):
        x =[]
        for j in iter:
            x.append(testX[i][j])
        x = numpy.array(x)
        newTestX.append(x)
    return newTrainX,newTestX

def addBais(trainX, testX):
    for i in range (len(trainX)):
        trainX[i] = numpy.append(trainX[i],[1])
    for i in range (len(testX)):
        testX[i] = numpy.append(testX[i],[1])
    return trainX,testX

def normilaisData(trainX,testX):
    numOffeathers = len(trainX[0])
    for i in range(numOffeathers):
        minValue = 0;
        maxValue = 0
        #find the min and the max for i feather
        for j in range(len(trainX)):
            if minValue> trainX[j][i]:
                minValue = trainX[j][i]
            if maxValue< trainX[j][i]:
                maxValue = trainX[j][i]
        for j in range(len(testX)):
            if minValue> testX[j][i]:
                minValue = testX[j][i]
            if maxValue< testX[j][i]:
                maxValue = testX[j][i]
        #update the feather i in all the data
        if maxValue != minValue:
            for j in range(len(trainX)):
                #trainX[j][i] = (trainX[j][i] - ((maxValue - minValue)/2) - minValue)/(maxValue-minValue)
                trainX[j][i] = (trainX[j][i]  - minValue) / (maxValue - minValue)
            for j in range(len(testX)):
                #testX[j][i] = (testX[j][i] - ((maxValue - minValue)/2) - minValue)/(maxValue-minValue)
                testX[j][i] = (testX[j][i] - minValue) / (maxValue - minValue)
    return trainX,testX

def checkAlgo(trainX, trainY,optimalK,bestlerningRatePerceptron, bestlerningRatePA):
    numOfErrorKnn=[]
    numOfErrorKnn2=[]
    numOfErrorPerceptron = []
    numOfErrorPa = []


    for i in range(1,100):
        tempTrainX, tempTrainY, tempTestData, tempTestAns = splitTrain(trainX, trainY, 0.9)
        Knn_yhat = KNNnerest(tempTrainX, tempTrainY, optimalK, tempTestData);
        Knn2_yhet = KNN2(tempTrainX, tempTrainY, optimalK, tempTestData)
        Perceptron_yhat = Perceptron(tempTrainX, tempTrainY, tempTestData, 3, 30,0.01)
        PA_yhat = PAalgo(tempTrainX, tempTrainY, tempTestData, 3,34,1)
        mistakeKnn2=0
        mistakeKnn = 0
        mistakePerceptron = 0
        mistakePa = 0
        for j in range(len(tempTestAns)):
            if Knn2_yhet[j] != tempTestAns[j]:
                mistakeKnn2 += 1
            if Knn_yhat[j] != tempTestAns[j]:
                mistakeKnn += 1
            if Perceptron_yhat[j] != tempTestAns[j]:
                mistakePerceptron += 1
            if PA_yhat[j] != tempTestAns[j]:
                mistakePa += 1
        numOfErrorKnn2.append(mistakeKnn2 / len(tempTestAns))
        numOfErrorKnn.append(mistakeKnn / len(tempTestAns))
        numOfErrorPerceptron.append(mistakePerceptron / len(tempTestAns))
        numOfErrorPa.append(mistakePa / len(tempTestAns))
    #plt.plot(range(1, 40), numOfErrorKnn, color='blue', linestyle='dashed',
    #         marker='o', markerfacecolor='red', markersize=10 ,label='Knn')
    #plt.plot(range(1, 40), numOfErrorPerceptron, color='green', linestyle='dashed',
    #         marker='o', markerfacecolor='red', markersize=10 ,label='Perceptron')
    #plt.plot(range(1, 40), numOfErrorPa, color='red', linestyle='dashed',
    #         marker='o', markerfacecolor='red', markersize=10 ,label='PA')
    #plt.title('Error Rate vs. Itaration Number')
    #plt.xlabel('Itaration Number')
    #plt.ylabel('Error Rate')
    print("Knn Average = "+str(numpy.average(numOfErrorKnn)) + "Knn2 Aveage = "+str(numpy.average(numOfErrorKnn2))+" Perceptron Average = "+str(numpy.average(numOfErrorPerceptron))+" PA Average = "+str(numpy.average(numOfErrorPa)))
    #plt.show()
    return  numpy.average(numOfErrorKnn),numpy.average(numOfErrorPerceptron),numpy.average(numOfErrorPa)
def findEpochOptimal(trainX,trainY,testX,numOfClassters):
    numOfErrorPa = []
    for i in range(0, 200):
        tempTrainX, tempTrainY, tempTestData, tempTestAns = splitTrain(trainX, trainY, 0.9)
        for w in range(50):
            PA_yhat = PAalgo(tempTrainX, tempTrainY, tempTestData, numOfClassters, w, 1)
            mistakePa = 0
            for j in range(len(tempTestAns)):
                if PA_yhat[j] != tempTestAns[j]:
                    mistakePa += 1
            if(i==0):
                numOfErrorPa.append(mistakePa / len(tempTestAns))
            else:
                numOfErrorPa[w]+=mistakePa / len(tempTestAns)
    for w in range(50):
        numOfErrorPa[w] = numOfErrorPa[w]/200
    print("the optimal cpoch is" + str(numOfErrorPa.index(min(numOfErrorPa))))

def PAalgo(trainX,trainY,testX,numOfClassters,epoch,lerningRate):
    predict=[]
    classtersVector = [[((random.random()-0.5)*2) for i in range(len(trainX[0]))] for j in range(numOfClassters)];
    zipTrain = zip(trainX, trainY)
    # zipTrain = shuffle(zip(trainX,trainY))
    for k in range(epoch):
        for x, y in zipTrain:
            yhat = getclosestClasster(classtersVector, x)
            if yhat == y:
                yhat = getSecondclosestClasster(classtersVector,x)
                tau = culcolateTau(classtersVector, x, y, yhat)
                classtersVector[yhat] = classtersVector[yhat] - lerningRate * tau * x;
                classtersVector[y] = classtersVector[y] + lerningRate * tau * x
            else:
                tau = culcolateTau(classtersVector,x,y,yhat)
                classtersVector[yhat] = classtersVector[yhat] - lerningRate*tau*x;
                classtersVector[y] = classtersVector[y] + lerningRate*tau*x
    for i in range(len(testX)):
        yhat = getclosestClasster(classtersVector,testX[i])
        predict.append(yhat)
    return predict

def culcolateTau(w,x,y,yhat):
    temp= 1 - numpy.dot( w[y] ,x) + numpy.dot(w[yhat] ,x)
    l = max(0,temp)
    div = 2* (numpy.linalg.norm(x)**2)
    if div !=0:
        return l/div
    else:
        return 0

def Perceptron(trainX,trainY,testX,numOfClassters, epoch,lerningRate):
    predict = []
    numofcurrect =0
    classterPerceptron =[[0.8727338348157087, 0.09873260902021264, 0.37994701063113034, 0.08604734423267935, 0.5236164866773904, 0.7966904532804855, 0.4325999985156985, 0.015930395207944037, 0.06727731559594718, 0.2615938994249438, 0.3435666225903351, 0.286540098081536, 0.04266335348888117], [0.8229354541546609, 0.9722526194370779, 0.10765240748736271, 0.948796870267857, 0.7434412556556899, 0.028322399699690415, 0.5827737663970975, 0.6377927217527938, 0.8716291210549922, 0.5688957433250615, 0.5310135018741589, 0.20434643486462312, 0.9076602610982552], [0.341843453351223, 0.6642004856314956, 0.7142719113551897, 0.4085959428848456, 0.08112617650054532, 0.06324651691215377, 0.5955994999723474, 0.799971703731538, 0.1632770937532595, 0.22542585319854258, 0.20252378581562813, 0.9178315075496001, 0.45666158667446355]]
        #[[((random.random()-0.5)*2) for i in range(len(trainX[0]))] for j in range(numOfClassters)];
    for j in range(epoch):
        numofcurrect=0
        zipTrain = zip(trainX,trainY)
        #zipTrain = shuffle(zip(trainX,trainY))
        for x,y in zipTrain:
            yhat = getclosestClasster(classterPerceptron,x)
            if yhat == y:
                numofcurrect+=1
                continue;
            else:
                classterPerceptron[yhat] = classterPerceptron[yhat] - lerningRate*x;
                classterPerceptron[y] = classterPerceptron[y] + lerningRate*x
        #print(numofcurrect)
    #print(classterPerceptron)
    for i in range(len(testX)):
        yhat = getclosestClasster(classterPerceptron,testX[i])
        predict.append(yhat)
    return predict

def getclosestClasster(classterPerceptron,x):
    tempArray =[]
    for i in range(len(classterPerceptron)):
        dis = sum(classterPerceptron[i]*x)
        tempArray.append(dis)
    return tempArray.index(max(tempArray))
def getSecondclosestClasster(classterPerceptron,x):
    tempArray =[]
    for i in range(len(classterPerceptron)):
        dis = sum(classterPerceptron[i]*x)
        tempArray.append(dis)
    tempArray.remove(max(tempArray))
    return tempArray.index(max(tempArray))
#knn function

def gettestAndDataset():
    file_trainX = open(sys.argv[1], "r")
    file_trainY = open(sys.argv[2], "r")
    file_testX = open(sys.argv[3], "r")
    trainX = file_trainX.readlines()
    for i in range(len(trainX)):
        trainX[i] = trainX[i].split(",")
        if trainX[i][11] == 'W\n':
            trainX[i][11] = 0
            #trainX[i].append(0)

        else:
            trainX[i][11] = 1
            #trainX[i].append(1)
        trainX[i] = numpy.array(trainX[i],float)

    trainY = file_trainY.readlines()
    trainY = [i.strip("\n") for i in trainY]
    trainY = numpy.array(trainY, numpy.int16)

    testX = file_testX.readlines()
    for i in range(len(testX)):
        testX[i] = testX[i].split(",")
        #testX[i] = numpy.array(testX[i], float)
        if testX[i][11] == 'W\n':
            testX[i][11] = 1
            #testX[i].append(0)
        else:
            testX[i][11] = -1
            #testX[i].append(1)
        testX[i] = numpy.array(testX[i], float)
    return trainX,trainY,testX
def KNNnerest(trainX,trianY, k, newData):
    ans =[]
    for j in range (len(newData)):
        neighbors = []
        for i in range(len(trainX)):
            #dis = distanse(trainX[i],newData[j])
            dis = dista.euclidean(trainX[i],newData[j])
            neighbors= update_neighbors(neighbors,k,[trainX[i],trianY[i]],dis);

        classter = culcolateClassterFromNeighbords(neighbors)
        ans.append(classter)
    return ans

def KNN2(trainX,trianY, k, newData):
    ans = []
    for j in range(len(newData)):
        distanseArray = []
        for i in range(len(trainX)):
            # dis = distanse(trainX[i],newData[j])
            dis = dista.euclidean(trainX[i], newData[j])
            distanseArray.append([dis,trianY[i]])
        distanseArray=sorted(distanseArray, key=isBigger)
        knirest = distanseArray[:k]
        classter =[0,0,0]
        for w in range(len(knirest)):
            classter[knirest[w][1]]+=1;
        classify = classter.index(max(classter))

        #classter = culcolateClassterFromNeighbords(neighbors)
        ans.append(classify)
    return ans

def culcolateClassterFromNeighbords(neighbors):
    counter ={}
    for i in range (len(neighbors)):
        if neighbors[i][1] not in counter:
            counter[neighbors[i][1]] = 1
        else:
            counter[neighbors[i][1]]+=1
    maxIndex=0;
    maxCounter =-1;
    for i in counter:
        if counter[i]> maxCounter:
            maxCounter= counter[i]
            maxIndex = i;
    return maxIndex
def distanse(x,y):
    sum = 0
    for i in range(len(x)):
        sum += math.pow(x[i]-y[i],2)
    return math.sqrt(sum);
def update_neighbors(neighbord, k, x,distanse):
    if len(neighbord)<k:
        neighbord.append([distanse,x[1]])
    else:
        if neighbord[k-1][0] > distanse:
            neighbord[k-1] = [distanse,x[1]]
    neighbord = sorted(neighbord, key=isBigger)
    return neighbord
def isBigger(x):
   return x[0]
def findOptimalK(trainX, trainY):
    numberOfMistake=[]
    for k in range(100):
        tempTrainX, tempTrainY, tempTestData, tempTestAns = splitTrain(trainX, trainY, 0.8)
        for i in range(35):
            avarageRate = 0
            predict = KNNnerest(tempTrainX,tempTrainY,i+1,tempTestData)
            mistake = 0
            for j in range(len(predict)):
                if predict[j] != tempTestAns[j]:
                    mistake+=1
            avarageRate += mistake/len(predict)
            if k ==0:
                numberOfMistake.append(avarageRate)
            else:
                numberOfMistake[i] += avarageRate
            print("iter" +str(i) +"in the "+str(k)+"iter")
    for k in range(35):
        numberOfMistake[k] = numberOfMistake[k]/40
    plt.plot(range(1, 36), numberOfMistake, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    print("Minimum error:-", min(numberOfMistake), "at K =", numberOfMistake.index(min(numberOfMistake))+1)
    return numberOfMistake.index(min(numberOfMistake))+1

def splitTrain(trainX,trainY,rate):
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
    return tempTrainX,tempTrainY,tempTestData,tempTestAns





if __name__ == '__main__':
    main()
