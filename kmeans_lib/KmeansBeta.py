# -*- coding: utf-8 -*-
# !/usr/bin/env python
from numpy import *
from kmm import *
import math
import scipy.io as sio
from kmeans import *
import numpy as np
from kmeans_plus_plus import *


def readStreamData(dataName,rate,index):
    sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_source_stream.txt'
    targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) +'_target_stream.txt'
    #sourcePath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) + '_source_stream.txt';
    #targetPath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) + '_target_stream.txt';

    sourceFeature = []
    targetFeature = []
    sourceLabel = []
    targetLabel = []

    with open(sourcePath) as file:
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]
            sourceFeature.append(process[:-1])
            sourceLabel.append(process[-1])

    with open(targetPath) as file:
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]
            targetFeature.append(process[:-1])
            targetLabel.append(process[-1])

    #print "after read, the length of source is: "+str(len(sourceFeature))+" and target length is: "+str(len(targetFeature))
    return sourceFeature,targetFeature,sourceLabel,targetLabel


def getNovelClass(tranTargetFeature,targetLabel):
    real_outlier = []
    tranTarget = []
    for index, point in enumerate(targetLabel):
        if targetLabel[index] == 1.0 or targetLabel[index] == 6.0 or targetLabel[index] == 7.0:
            tranTarget.append(tranTargetFeature[index])
        else:
            real_outlier.append(tranTargetFeature[index])

    return real_outlier,tranTarget



# def readSourceData(dataName,d,rate,index):
#     sfPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_d=' + str(
#         d) + '.mat';
#
#     data = sio.loadmat(sfPath)
#     sourceLabel = data['sourceLabel']
#     targetLabel = data['targetLabel']
#     sourceFeature = data['sourceFeature']
#     targetFeature = data['targetFeature']
#
#     return sourceLabel,targetLabel,sourceFeature,targetFeature


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

def initPlusCentroids(dataSet,K):
    centroids = get_centroids(mat(dataSet), K)
    return centroids

# def computeMeanBeta(indexInClust,betaValue):
#     sum = 0.0
#     for i in indexInClust:
#         sum+=betaValue[i]
#
#     return sum/size(indexInClust)

def computeSigma(pointsInClust):
    #sigmaValue = 0.0
    sum = 0.0
    sigmaValue = var(pointsInClust)
    # print "sigmaArray",sigmaArray
    # arrayLength = len(sigmaArray)*1.0
    # for i in sigmaArray:
    #     sum+=sigmaArray[i]
    #
    # sigmaValue = sum/arrayLength
    return 1/sqrt(sigmaValue)

def computeRadius(indexInClust,dataSet,centerPoint):
    sum = 0.0
    max = 0.0
    for i in indexInClust:
        sum+=calcDist(dataSet[i,:],centerPoint)

    return sum/size(indexInClust)

def computeSigmaRadius(indexInClust, dataSet, centerPoint):
    sum = 0.0
    c = 2.5 # times of the variance
    distanceList = []
    for i in indexInClust:
        distanceInItem = euclDistance(dataSet[i, :], centerPoint)
        distanceList.append(power(distanceInItem,1))
        # maxR = max(maxR,euclDistance(dataSet[i,:],centerPoint))

    meanDis = mean(distanceList)
    varDis = var(distanceList)

    return meanDis + c * varDis


def computeBetaRadius(indexInClust, dataSet, centerPoint,betaClusValue):
    sum = 0.0
    c = 2.0
    distanceList = []
    for i in indexInClust:
        distanceBeta= euclDistance(dataSet[i, :], centerPoint)*betaClusValue
        distanceList.append(distanceBeta)

    meanDis = mean(distanceList)
    varDis = var(distanceList)
    return meanDis + c * varDis

# def kMeansBeta(dataSet, k, betaValue, distMeas=calcDist, createCent=initCentroids):
#
#     m = shape(dataSet)[0]
#     clusterAssment = mat(zeros((m,2)))#create mat to assign data points
#                                       #to a centroid, also holds SE of each point
#     centroids = createCent(dataSet, k)
#     #centroids,betaCenter = createCent(dataSet, k,betaValue)
#     clusterChanged = True
#     radiusCluster = [0] * k
#     betaCluster = [0] * k
#     #sigmaCluster = [1] * k
#     while clusterChanged:
#         clusterChanged = False
#         for i in range(m):#for each data point (0,m) assign it to the closest centroid
#             minDist = inf
#             minIndex = -1
#             for j in range(k):
#                 distJI = calcDist(centroids[j,:],dataSet[i,:])
#                 #print distJI
#                 #distBeta = fabs(betaCenter[j] - betaValue[i])
#                 if betaValue[i]*distJI < minDist: #minimize distance+beta decreasing power(Vec1 - Vec2, 2)
#                 #if power(distJI, 2) + power(distBeta,2) < minDist:
#                     minDist = betaValue[i]*distJI
#                     minIndex = j
#             if clusterAssment[i,0] != minIndex:
#                 clusterChanged = True
#             clusterAssment[i,:] = minIndex,minDist**2
#
#
#         for j in range(k):#recalculate centroids
#             indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
#             #print "The size of cluster"+str(j)+"in EM is:",size(indexInClust)
#             #betaCenter[j] = computeMeanBeta(indexInClust,betaValue)
#
#             ptsInClust = dataSet[indexInClust]#get all the point in this cluster
#             centroids[j,:] = mean(ptsInClust, axis=0) #assign centroid to mean
#
#             betaClusSum = 0.0
#             count = 0.0
#             for i in indexInClust:
#                 betaClusSum+=betaValue[i]
#                 count+=1
#             betaCluster[j] = betaClusSum/count
#
#             radiusCluster[j] = computeBetaRadius(indexInClust, dataSet, centroids[j, :], betaCluster[j])
#
#             #radiusCluster[j] = computeSigmaRadius(indexInClust, dataSet, centroids[j, :])
#
#
#     print "========================================================================================================="
#
#     return centroids,clusterAssment,radiusCluster,betaCluster
#     #return centroids, clusterAssment, radiusCluster



def kMeansBeta(ori_dataSet, k, start, end, betaValue, distMeas=calcDist, createCent=initCentroids):

    #m = shape(dataSet)[0]
    m = end - start
    dataSet = ori_dataSet[start:end]
    #m = 100000
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    #centroids,betaCenter = createCent(dataSet, k,betaValue)
    clusterChanged = True
    radiusCluster = [0] * k
    betaCluster = [0] * k
    realIndexClus = {}
    #sigmaCluster = [1] * k
    i = start
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point (0,m) assign it to the closest centroid
        #for i in range(start,end):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = calcDist(centroids[j,:],dataSet[i,:])
                #print distJI
                #distBeta = fabs(betaCenter[j] - betaValue[i])
                if betaValue[i]*distJI < minDist: #minimize distance+beta decreasing power(Vec1 - Vec2, 2)
                #if power(distJI, 2) + power(distBeta,2) < minDist:
                    minDist = betaValue[i]*distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2


        for j in range(k):#recalculate centroids
            indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
            for index in indexInClust:
                realIndexClus[index] = j

            ptsInClust = dataSet[indexInClust]#get all the point in this cluster
            centroids[j,:] = mean(ptsInClust, axis=0) #assign centroid to mean

            betaClusSum = 0.0
            count = 0.0
            for i in indexInClust:
                betaClusSum+=betaValue[i]
                count+=1
            betaCluster[j] = betaClusSum/count

            radiusCluster[j] = computeBetaRadius(indexInClust, dataSet, centroids[j, :], betaCluster[j])
           #radiusCluster[j] = computeSigmaRadius(indexInClust, dataSet, centroids[j, :])
    betaSet = {}
    i = start
    for betaItem in betaValue:
        betaSet[i] = betaItem
        i += 1

    print "========================================================================================================="

    return centroids,realIndexClus,radiusCluster,betaCluster,betaSet
    #return centroids, clusterAssment, radiusCluster



def Betapredict(dataSet,centroids,k,radiusCluster,betaCluster):
    m = dataSet.shape[0] #number of the dataSet
    count = 0.0
    for i in range(m):
        #everyCluster = 0.0
        #print "i: ",i
        outlier = True
        for j in range(k):
            distance = calcDist(dataSet[i,:],centroids[j,:])*betaCluster[j]
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

# def BetacalcPurity(clusterAssment,k,sourceLabel):
#     purity = []
#     for j in range(k):
#         #everyInfo = 0.0
#         maxPoint = 0.0
#         minPoint = 0.0
#         labelSet = {}
#         indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
#
#         total = len(indexInClust)*1.0
#         print "In"+str(j)+" th cluster, the total point is",total
#         for i in indexInClust:
#             pointLabel = sourceLabel[i]
#             if pointLabel not in labelSet.keys():
#                 labelSet[pointLabel] = 0
#
#             labelSet[pointLabel]+=1
#
#         for key in labelSet:
#             maxPoint = max(maxPoint,labelSet[key])
#
#         print "maxLabel number", maxPoint
#         purity.append(maxPoint/total)
#
#     return purity



if __name__ == '__main__':
    gammab = [10.0]
    dataName = "Syndata_002"
    rate = 0.3
    K = 5
    index = 1
    novelClassList = []
    start = 100
    end = 1800

    sourceFeature, targetFeature, sourceLabel, targetLabel = readStreamData(dataName,rate,index)
    print "source data length: ",len(sourceFeature)
    print "target data length: ",len(targetFeature)

    beta_sourceFeature = sourceFeature[start:end]
    betaValue = getBeta(beta_sourceFeature, targetFeature, gammab)
    print "length of beta value: ",len(betaValue)

    #tranSourceFeature = transform(sourceFeature,betaValue)
    #print "tranSourceFeature: ",tranSourceFeature

    centroids, realIndexClus, radiusCluster, betaCluster, betaSet = kMeansBeta(mat(sourceFeature), K, start, end, betaValue)
    real_outlier, tranTarget = getNovelClass(targetFeature, targetLabel)

    # purity = BetacalcPurity(clusterAssment,K,sourceLabel)
    # print "purity",purity
    # #average
    # sumPurity = 0.0
    # for i in purity:
    #     sumPurity+=i
    # print "avg purity",sumPurity/len(purity)

    print "For real label: "
    Betapredict(mat(tranTarget),centroids, K, radiusCluster, betaCluster)

    print "For novel class: "
    Betapredict(mat(real_outlier), centroids, K, radiusCluster, betaCluster)


