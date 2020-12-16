import sys

import numpy
import scipy
import scipy.io.wavfile
import numpy as np


def main(st1, str2):
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    cent = centroids.copy()
    centroidsNew = [[] for i in range(len(centroids))]
    for i in range(len(centroids)):
        centroidsNew[i] = [centroids[i][0], centroids[i][1]]

    file = open("output.txt", "w")

    i = 0
    for k in range(30):
        counter = 0
        clusters = assignToCentroid(x, centroidsNew)
        for i in range(len(clusters)):
            dx = 0
            dy = 0
            for j in range(len(clusters[i])):
                dx = dx + clusters[i][j][0]
                dy = dy + clusters[i][j][1]
            len_arr = len(clusters[i])
            midX = round(dx / len_arr)
            midY = round(dy / len_arr)

            newCentroid = [midX, midY]
            if centroidsNew[i] == newCentroid:
                counter = counter + 1
            else:
                centroidsNew[i] = newCentroid
        for i in range(len(cent)):
            cent[i][0] = centroidsNew[i][0]
            cent[i][1] = centroidsNew[i][1]

        file.write(f"[iter {k}]:{','.join([str(r) for r in cent])}\n")
        if counter == len(clusters):
            break
    file.close()
    s = 2


def assignToCentroid(pointsArray, centroids):
    centeredArray = [[] for i in range(len(centroids))]
    for i in range(len(pointsArray)):
        minIndex = 0
        minDist = calculateDistance(pointsArray[i], centroids[0])
        for j in range(len(centroids)):
            dist = calculateDistance(pointsArray[i], centroids[j])
            if dist < minDist:
                minDist = dist  # assign the min distance to continue checking
                minIndex = j
        centeredArray[minIndex].append(pointsArray[i])
    return centeredArray


def calculateDistance(point, centroid):
    dist = numpy.linalg.norm(point - centroid)
    return dist


main(sys.argv[1], sys.argv[2])
