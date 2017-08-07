# -*- coding: utf-8 -*-
# !/usr/bin/env python
from numpy import *
from kmm import *
import math
import scipy.io as sio
from kmeans import *
import numpy as np
from sklearn.cluster import KMeans
from kmeans_plus_plus import *
from Novel_Class.clusteringMethod import *
from sklearn import svm


def readBetaStreamData(dataName,rate,index):
    #sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_source_stream.txt'
    #targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) +'_target_stream.txt'
    sourcePath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) + '_source_stream.txt';
    targetPath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) + '_target_stream.txt';

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

def computeRadius(indexInClust,dataSet,centerPoint,betaClusValue):
    sum = 0.0
    maxR = 0.0
    for i in indexInClust:
        curDist = euclDistance(dataSet[i, :], centerPoint)*betaClusValue
        maxR = max(maxR,curDist)

    #return sum/size(indexInClust)
    return maxR

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
    c = 2.5
    distanceList = []
    for i in indexInClust:
        distanceBeta = euclDistance(dataSet[i, :], centerPoint)*betaClusValue
        distanceList.append(distanceBeta)

    meanDis = mean(distanceList)
    varDis = var(distanceList)

    return meanDis + c * varDis

# k-means cluster
def ori_kmeans(ori_dataSet, k,start,end,souceLabel):
    dataSet = ori_dataSet[start:end]
    numSamples = dataSet.shape[0]  # 行数
    clusRealLabel = [0] * k
    clusterAssment = mat(zeros((numSamples, 2)))  #
    realIndexSet = {}
    clusterChanged = True  # 停止循环标志位

    ## step 1: init 初始化k个质点
    centroids = initCentroids(dataSet, k)
    radiusCluster = [0] * k
    while clusterChanged:
        clusterChanged = False
        ## for each 行
        for i in xrange(numSamples):
            minDist = 100000.0  # 设定一个极大值
            minIndex = 0
            ## for each centroid
            ## step 2: 寻找最接近的质心
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                # 将centroids（k个初始化质心）的j行和dataset（数据全集）的i行 算欧式距离，返回数值型距离
                if distance < minDist:
                    # 找距离最近的质点，记录下来。
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster # 跟新这个簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # clusterAssment 是一个n行2列的矩阵  Assment 评估
                clusterAssment[i, :] = minIndex, minDist ** 2  # 赋值为 新的质点标号

        ## step 4: update centroids
        for j in range(k):
            # 属于j这个质点的所有数值的平均值算出成为新的质点
            indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
            pointsInCluster = dataSet[indexInClust]
            centroids[j, :] = mean(pointsInCluster, axis=0)
            # caculate the radius of every cluster
            radiusCluster[j] = computeSigmaRadius(indexInClust, dataSet, centroids[j, :])

    print 'Congratulations, cluster complete!'

    # assign clusterID to real index
    realIndexClus = {}
    index = start
    for assignItem in clusterAssment[:, 0].tolist():
        # print assignItem[0]
        realIndexClus[index] = assignItem[0]
        index += 1
    # print "real index length: ",len(realIndexClus)
    print "========================================================================================================="
    # assign real label
    for j in range(k):
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
        clusRealLabel[j] = clusLabel(indexInClust, sourceLabel)

    return centroids, realIndexClus, radiusCluster,clusRealLabel

def clusLabel(indexInClust,sourceLabel):
    labelSet = {}
    maxPoint = 0.0
    clusAssignLabel = 0
    for item in indexInClust:
        pointLabel = sourceLabel[item]
        if pointLabel not in labelSet.keys():
            labelSet[pointLabel] = 0

        labelSet[pointLabel] += 1

    for key in labelSet:
        if labelSet[key] > maxPoint:
            maxPoint = labelSet[key]
            clusAssignLabel = key

    return clusAssignLabel

def kMeansBeta(dataSet, k, betaValue, sourceLabel,distMeas=calcDist, createCent=initCentroids):

    m = shape(dataSet)[0]
    #m = end - start
    #dataSet = ori_dataSet[start:end]
    #m = 100000
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points to a centroid, also holds SE of each point
    clusterPoints = [[]]*k  # create array to store data points
    centroids = createCent(dataSet, k)
    #centroids,betaCenter = createCent(dataSet, k,betaValue)
    clusterChanged = True
    radiusCluster = [0] * k
    betaCluster = [0] * k
    clusRealLabel = [0] * k

    #sigmaCluster = [1] * k
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
            ptsInClust = dataSet[indexInClust]#get all the point in this cluster
            clusterPoints[j] = ptsInClust
            centroids[j,:] = mean(ptsInClust, axis=0) #assign centroid to mean

            betaClusSum = 0.0
            count = 0.0
            for i in indexInClust:
                betaClusSum+=betaValue[i]
                count+=1
            if(count != 0):
                betaCluster[j] = betaClusSum/count
            else:
                betaCluster[j] = 0.0

            #radiusCluster[j] = computeBetaRadius(indexInClust, dataSet, centroids[j, :], betaCluster[j])
            radiusCluster[j] = computeRadius(indexInClust, dataSet, centroids[j, :],betaCluster[j])


    print "========================================================================================================="
    #assign real label
    for j in range(k):
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
        clusRealLabel[j] = clusLabel(indexInClust, sourceLabel)

    return centroids,clusterPoints,radiusCluster,clusRealLabel,betaCluster
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

def normalPredict(dataSet,centroids,k,radiusCluster):
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


def assignLabel(targetPoint,centroids,clusRealLabel):
    minDisToClus = 1000000000.0000
    minClusID = 0.0

    k = len(centroids)
    for j in range(k):
        curDis = euclDistance(targetPoint, centroids[j, :])
        if curDis < minDisToClus:
            minClusID = j
            minDisToClus = curDis

    predictLabel = clusRealLabel[minClusID]
    return predictLabel


if __name__ == '__main__':
    gammab = [32.0]
    dataName = "Syndata_002"
    rate = 0.3
    K = 8
    index = 1
    novelClassList = []
    start = 0
    end = 500

    sourceFeature, targetFeature, sourceLabel, targetLabel = readBetaStreamData(dataName,rate,index)
    targetFeature = targetFeature[start:end * 2]
    targetLabel = targetLabel[start:end * 2]
    print "source data length: ",len(sourceFeature)
    print "target data length: ",len(targetFeature)

    #beta_sourceFeature = sourceFeature[start:end]
    betaValue = getBeta(sourceFeature, targetFeature, gammab)
    print "length of beta value: ",len(betaValue)

    #tranSourceFeature = transform(sourceFeature,betaValue)
    #print "tranSourceFeature: ",tranSourceFeature

    centroids, clusterPoints, radiusCluster, clusRealLabel, betaCluster = kMeansBeta(mat(sourceFeature), K, betaValue,sourceLabel)
    #print "K number: ",len(centroids)
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

    print "========================================================================"
    all_test = len(targetLabel)
    right = 0.0
    for index,point in enumerate(targetFeature):
        predictLabel = assignLabel(point,centroids,clusRealLabel)
        if(predictLabel == targetLabel[index]):
            right+=1

    print "whole predict target point is: ",all_test
    print "Accuracy: ",right/all_test


    # kmeans algorithm
    kmeans = KMeans(n_clusters=8, random_state=0).fit(sourceFeature)
    labels = kmeans.labels_
    clusId2label = clus_Info(labels, 8, sourceLabel)
    predict_same = kmeans.predict(targetFeature)

    count_label = 0.0
    predict_tran = np.ones(shape=len(predict_same), dtype=int) * -1
    for index, predictId in enumerate(predict_same):
        #print "clusId2label[predictId]: ",clusId2label[predictId]
        predict_tran[index] = clusId2label[predictId]
        if predict_tran[index] == targetLabel[index]:
            count_label+=1

    print "len of same label target: ",len(targetLabel)
    print count_label/len(targetLabel)

    #svm
    # svc = svm.SVC(C = 20.0, kernel = 'rbf',gamma = 0.1)
    # svc.fit(np.array(sourceFeature), sourceLabel, sample_weight=None)
    # pre = svc.predict(np.array(targetFeature))
    # acc = float((pre == targetLabel).sum()) / len(targetLabel)
    # print "accuracy is: ",acc
