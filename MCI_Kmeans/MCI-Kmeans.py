# coding:UTF-8
from numpy import *
import math
import scipy.io as sio
import numpy as np

sourceData = []
sourceLabel = []
changedAtall = False  # any cluster changes after E_M?


def readTargetData(path):
    feature = []
    label = []
    # i = 0
    with open(path) as file:
        countA = 0
        countB = 0
        countC = 0
        countAll = 0
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]

            if process[-1] == 1.0 and countA != 400:
                countA += 1
                feature.append(process[:-1])
                label.append(process[-1])

            elif process[-1] == 2.0 and countB != 400:
                countB += 1
                feature.append(process[:-1])
                label.append(process[-1])
            elif process[-1] == 3.0 and countC != 400:
                countC += 1
                feature.append(process[:-1])
                label.append(process[-1])
                # i+=1
    return feature, label


def readTestData(path):
    feature = []
    label = []
    count = 0
    with open(path) as file:
        for rows in file:
            if count < 300:
                process = []
                line = rows.split(',')
                process = [float(x) for x in line]
                # print process[-1]
                if process[-1] == 5.0:
                    count += 1
                    feature.append(process[:-1])
                    label.append(process[-1])
    return feature, label


def calcDist(Vec1, Vec2):
    return sqrt(sum(power(Vec2 - Vec1, 2)))


# 初始化质心随机样本
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape #获取数据集合的行列总数
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        # uniform() 方法将随机生成下一个实数，它在[x,y]范围内。
        centroids[i, :] = dataSet[index, :]
    return centroids


def computeRadius(indexInClust, dataSet, centerPoint):
    sum = 0.0
    max = 0.0
    for i in indexInClust:
        sum += calcDist(dataSet[i, :], centerPoint)

    return sum / size(indexInClust)

def computeSigmaRadius(indexInClust,dataSet,centerPoint):
    sum = 0.0
    c = 2 #times of the variance
    distanceList = []
    for i in indexInClust:
        distanceList.append(calcDist(dataSet[i,:],centerPoint))
        #maxR = max(maxR,euclDistance(dataSet[i,:],centerPoint))

    meanDis = mean(distanceList)
    varDis = var(distanceList)

    return meanDis+c*varDis


def calcEntropy(clusterAssment, k, sourceLabel):
    sumEntropy = 0.0
    for j in range(k):
        KthEntropy = 0.0;
        labelSet = {}
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]

        total = len(indexInClust) * 1.0
        print "In" + str(j) + " th cluster, the total point is", total
        for i in indexInClust:
            pointLabel = sourceLabel[i]
            if pointLabel not in labelSet.keys():
                labelSet[pointLabel] = 0

            labelSet[pointLabel] += 1

        for key in labelSet:
            # maxPoint = max(maxPoint,labelSet[key])
            Pi = labelSet[key] / total
            KthEntropy += (-Pi) * math.log(Pi, 2)
            print "In" + str(j) + " th cluster, the cluster Entropy is", KthEntropy
        # purity.append(maxPoint/total)
        sumEntropy += KthEntropy

    return sumEntropy


# def ADCfuntion(clusterAssment,k,sourceLabel):

def calcObject(clusterId, pointId, sourceLabel, clusterAssment):
    entropy = 0.0
    otherLabel = 0.0
    labelSet = {}
    indexInClust = nonzero(clusterAssment[:, 0].A == clusterId)[0]
    total = len(indexInClust) * 1.0

    print "length of cluster"+str(clusterId)+" is: ",total

    # calculate frequency of labels in cluster
    for i in indexInClust:
        pointLabel = sourceLabel[i][0]
        if pointLabel not in labelSet.keys():
            labelSet[pointLabel] = 0

        labelSet[pointLabel] += 1

    pi = [0] * (len(labelSet)+1)

    for key in labelSet:
        if sourceLabel[pointId][0] == key:
            pi[key] = labelSet[key] / total
        else:
            pi[key] = labelSet[key] / total
            otherLabel += labelSet[key]

    # calculate entropy
    for key in labelSet:
        if pi[key] > 0:
            entropy -= (pi[key]) * math.log(pi[key], 2)

    return entropy * otherLabel



def MCL_Kmeans(dataSet, k, sourceLabel, distMeas=calcDist, createCent=initCentroids):
    numSamples = dataSet.shape[0]  # 行数
    clusterAssment = mat(zeros((numSamples, 2)))  #
    clusterChanged = True  # 停止循环标志位

    ## step 1: init 初始化k个质点
    centroids = initCentroids(dataSet, k)
    radiusCluster = [0] * k
    while clusterChanged:
        iteration = 0
        clusterChanged = False
        ## for each 行
        for i in xrange(numSamples):
            minObj = 100000000000000.0  # 设定一个极大值
            minIndex = 0
            ## for each centroid
            ## step 2: 寻找最接近的质心
            for j in range(k):
                distance = calcDist(centroids[j, :], dataSet[i, :])
                if iteration == 0:
                    objValue = distance
                else:
                    objValue = distance+distance*calcObject(j,i,sourceLabel,clusterAssment)
                if objValue < minObj:
                    minObj = objValue
                    minIndex = j

            ## step 3: update its cluster # 跟新这个簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # clusterAssment 是一个n行2列的矩阵  Assment 评估
                clusterAssment[i, :] = minIndex, minObj ** 2  # 赋值为 新的质点标号

        ## step 4: update centroids
        for j in range(k):
            # 属于j这个质点的所有数值的平均值算出成为新的质点
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)
            indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
            radiusCluster[j] = computeSigmaRadius(indexInClust, dataSet, centroids[j, :])
        iteration+=1;

    print 'Congratulations, cluster complete!'
    return centroids, clusterAssment, radiusCluster

def calcPurity(clusterAssment,k,sourceLabel):
    purity = []
    for j in range(k):
        #everyInfo = 0.0
        maxPoint = 0.0
        minPoint = 0.0
        labelSet = {}
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]

        total = len(indexInClust)*1.0
        print "In"+str(j)+" th cluster, the total point is",total
        for i in indexInClust:
            pointLabel = sourceLabel[i][0]
            if pointLabel not in labelSet.keys():
                labelSet[pointLabel] = 0

            labelSet[pointLabel]+=1

        for key in labelSet:
            maxPoint = max(maxPoint,labelSet[key])

        print "maxLabel number", maxPoint
        purity.append(maxPoint/total)

    return purity



def outlierPredict(dataSet, centroids, k, radiusCluster):
    outlierDist = []
    m = dataSet.shape[0]  # number of the dataSet
    count = 0.0
    for i in range(m):
        # everyCluster = 0.0
        # print "i: ",i
        outlier = True
        minDisClu = 100000000000000.0
        for j in range(k):
            distance = calcDist(dataSet[i, :], centroids[j, :])
            if distance <= radiusCluster[j]:
                outlier = False
                break
            else:
                #distBound = (distance - centroids[j,:])*1.0
                minDisClu = min(minDisClu, distance)

        # count+=1;
        if outlier == False:
            continue
        else:
            outlierDist.append(minDisClu)
            count += 1

    print "count", count
    print "m", m
    print "outlier length",len(outlierDist)
    print "outlier accuracy: ", count / m
    print "outlier mean",mean(outlierDist)
    print "outlier variance",var(outlierDist)

# def Non_outlierPredict(dataSet, centroids, k, radiusCluster):
#         outlierDist = []
#         m = dataSet.shape[0]  # number of the dataSet
#         count = 0.0
#         for i in range(m):
#             # everyCluster = 0.0
#             # print "i: ",i
#             outlier = True
#
#             for j in range(k):
#                 distance = calcDist(dataSet[i, :], centroids[j, :])
#                 if distance <= radiusCluster[j]:
#                     outlier = False
#                     break
#             # count+=1;
#
#             if outlier == False:
#                 continue
#             else:
#                 count += 1
#
#         print "count", count
#         print "m", m
#         print "outlier accuracy: ", count / m



if __name__ == '__main__':
    dataName = "Syndata_c5"
    rate = 0.5
    d = 20
    #sfPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(rate) + '.mat';
    sfPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(rate) + '_d=' + str(
        d) + '.mat';
    # Lpath = 'C:/Matlab_Code/infometric_0.1/Data/L.mat';
    targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(
        rate) + '_target_stream.txt'

    data = sio.loadmat(sfPath)
    # Matrix = sio.loadmat(Lpath)

    tranSourceFeature = data['tranSourceF']
    sourceLabel = data['sourceLabel']
    tranTargetFeature = data['trantargetF']
    targetLabel = data['targetLabel']
    transferMatrix = data['L']
    print "sourceLabel ", sourceLabel
    # original data
    sourceFeature = data['sourceFeature']
    targetFeature = data['targetFeature']

    print "source data's length: ", len(tranSourceFeature)

    testData, testLabel = readTestData(targetPath)
    print "target outlier length: ", len(testLabel)
    ori_testData = testData

    K = 8
    transferMatrix = np.array(transferMatrix).T
    testData = np.array(testData).T
    outlierTest = dot(transferMatrix, testData)
    outlierTest = outlierTest.T
    outlierTest.tolist()

    centroids, clusterAssment, radiusCluster = MCL_Kmeans(mat(tranSourceFeature), K, sourceLabel)

    # caculate purity
    purity = calcPurity(clusterAssment, K, sourceLabel)
    print "purity", purity

    sumPurity = 0.0
    for i in purity:
        sumPurity += i

    print "avg purity", sumPurity / len(purity)

    outlierPredict(mat(tranTargetFeature), centroids, K, radiusCluster)