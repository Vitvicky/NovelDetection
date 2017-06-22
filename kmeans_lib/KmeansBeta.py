# -*- coding: utf-8 -*-
# !/usr/bin/env python
from numpy import *
from kmm import *
import math
import scipy.io as sio
from kmeans import *

def getNovelClass(tranTargetFeature,targetLabel):
    real_outlier = []
    tranTarget = []
    for index, point in enumerate(targetLabel):
        if targetLabel[index][0] == 1.0 or targetLabel[index] == 6.0 or targetLabel[index] == 7.0:
            tranTarget.append(tranTargetFeature[index])
        else:
            real_outlier.append(tranTargetFeature[index])

    return real_outlier,tranTarget



def readSourceData(dataName,d,rate,index):
    sfPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_d=' + str(
        d) + '.mat';

    data = sio.loadmat(sfPath)
    sourceLabel = data['sourceLabel']
    targetLabel = data['targetLabel']
    sourceFeature = data['sourceFeature']
    targetFeature = data['targetFeature']

    return sourceLabel,targetLabel,sourceFeature,targetFeature


def transform(sourceData,betaValue):
    tranSourceFeature = []
    for dataIndex,dataItem in enumerate(sourceData):
        tranItem = dataItem*betaValue[dataIndex]
        tranSourceFeature.append(tranItem)
    print "transfer source data length is: ",len(tranSourceFeature)
    return tranSourceFeature


def readTestData(path):
    feature = []
    label = []
    count = 0
    with open(path) as file:
        for rows in file:
            if count<300:
                process = []
                line = rows.split(',')
                process = [float(x) for x in line]
                #print process[-1]
                if process[-1] == 4.0:
                    count+=1
                    feature.append(process[:-1])
                    label.append(process[-1])
    return feature,label


def calcDist(Vec1,Vec2):
    return sqrt(sum(power(Vec2 - Vec1, 2)))

def initCentroidsBeta(dataSet, k,betaValue):
    numSamples, dimension = shape(dataSet) #get the raw and col number of the dataset
    centroids = zeros((k, dimension))
    betaCenter = [0]*k
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        # uniform() 方法将随机生成下一个实数，它在[x,y]范围内。
        centroids[i] = dataSet[index]
        betaCenter[i] = betaValue[index]

    return centroids,betaCenter


def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


def computeMeanBeta(indexInClust,betaValue):
    sum = 0.0
    for i in indexInClust:
        sum+=betaValue[i]

    return sum/size(indexInClust)


def computeRadius(indexInClust,dataSet,centerPoint):
    sum = 0.0
    max = 0.0
    for i in indexInClust:
        sum+=calcDist(dataSet[i,:],centerPoint)

    return sum/size(indexInClust)

def computeSigmaRadius(indexInClust, dataSet, centerPoint):
    sum = 0.0
    c = 2.2  # times of the variance
    distanceList = []
    for i in indexInClust:
        distanceInItem = euclDistance(dataSet[i, :], centerPoint)
        distanceList.append(power(distanceInItem,1))
        # maxR = max(maxR,euclDistance(dataSet[i,:],centerPoint))

    meanDis = mean(distanceList)
    varDis = var(distanceList)

    return meanDis + c * varDis

def kMeansBeta(dataSet, k, betaValue, distMeas=calcDist, createCent=initCentroids):

    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    #centroids,betaCenter = createCent(dataSet, k,betaValue)
    clusterChanged = True
    radiusCluster = [0] * k
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point (0,m) assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = calcDist(centroids[j,:],dataSet[i,:])
                #print distJI
                #distBeta = fabs(betaCenter[j] - betaValue[i])
                if distJI*betaValue[i]  < minDist: #minimize distance+beta decreasing power(Vec1 - Vec2, 2)
                #if power(distJI, 2) + power(distBeta,2) < minDist:
                    minDist = distJI*betaValue[i]
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2


        for j in range(k):#recalculate centroids
            indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
            #print "The size of cluster"+str(j)+"in EM is:",size(indexInClust)
            #betaCenter[j] = computeMeanBeta(indexInClust,betaValue)

            ptsInClust = dataSet[indexInClust]#get all the point in this cluster
            centroids[j,:] = mean(ptsInClust, axis=0) #assign centroid to mean
            radiusCluster[j] = computeSigmaRadius(indexInClust, dataSet, centroids[j, :])
    print "========================================================================================================="

    return centroids, clusterAssment, radiusCluster

def Betapredict(dataSet,centroids,k,radiusCluster):
    m = dataSet.shape[0] #number of the dataSet
    count = 0.0
    for i in range(m):
        #everyCluster = 0.0
        #print "i: ",i
        outlier = True
        for j in range(k):
            distance = calcDist(dataSet[i,:],centroids[j,:])
            #print "redius of center "+str(j)+" is: ",radiusCluster[j]
            #print "distance of unknow label "+str(i)+" to center is: ",distance
            if distance<=radiusCluster[j]:
                outlier = False
                break
        #count+=1;

        if outlier == False:
            continue
        else:
            count+=1

    print "count",count
    print "m",m
    print "outlier accuracy: ", count/m

def BetacalcPurity(clusterAssment,k,sourceLabel):
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
            pointLabel = sourceLabel[i]
            if pointLabel not in labelSet.keys():
                labelSet[pointLabel] = 0

            labelSet[pointLabel]+=1

        for key in labelSet:
            maxPoint = max(maxPoint,labelSet[key])

        print "maxLabel number", maxPoint
        purity.append(maxPoint/total)

    return purity



if __name__ == '__main__':
    gammab = [0.001]
    dataName = "Syndata_002"
    rate = 0.3
    K = 5
    q = 2
    novelClassList = []
# sourcePath = 'C:/DataSet/Stream/Generate/dataset_set3_ori/'+str(fileName)+'/'+str(fileName)+'-bias_source_stream.csv'
# targetPath = 'C:/DataSet/Stream/Generate/dataset_set3_ori/'+str(fileName)+'/'+str(fileName)+'-bias_target_stream.csv'
# sourceData,sourceLabel = readSourceData(sourcePath)
# targerData,targetLabel = readTargetData(targetPath)

    sourceLabel, targetLabel, sourceFeature, targetFeature = readSourceData(dataName, 10, rate,2)
    print "source data length: ",len(sourceFeature)
    print "target data length: ",len(targetFeature)

    betaValue = getBeta(sourceFeature, targetFeature, gammab)
    print "length of beta value: ",len(betaValue)

    #tranSourceFeature = transform(sourceFeature,betaValue)
    #print "tranSourceFeature: ",tranSourceFeature

    centroids, clusterAssment, radiusCluster = kMeansBeta(mat(sourceFeature), K, betaValue)
    real_outlier, tranTarget = getNovelClass(targetFeature, targetLabel)

    purity = calcPurity(clusterAssment,K,sourceLabel)
    print "purity",purity
    #average
    sumPurity = 0.0
    for i in purity:
        sumPurity+=i
    print "avg purity",sumPurity/len(purity)

    predict(mat(real_outlier),centroids, K, radiusCluster)



