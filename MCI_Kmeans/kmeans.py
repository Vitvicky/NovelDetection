# -*- coding: utf-8 -*-
# !/usr/bin/env python

import csv
from numpy import *
import time
import matplotlib.pyplot as plt

#read data
def load_data(path):
    feature = []
    label = []
    i = 0
    with open(path) as file:
        countA = 0
        countB = 0
        countC = 0
        countAll = 0
        for rows in file:
                process = []
                line = rows.split(',')
                process = [float(x) for x in line]
                # if process[-1] == 0.0 or process[-1] == 1.0 or process[-1] == 2.0:
                #     feature.append(process[:-1])
                #     label.append(process[-1])
                #     countAll+=1
                if process[-1] == 1.0 and countA!=300:
                    countA+=1
                    feature.append(process[:-1])
                    label.append(process[-1])

                elif process[-1] == 2.0 and countB!=300:
                    countB += 1
                    feature.append(process[:-1])
                    label.append(process[-1])
                elif process[-1] == 3.0 and countC != 300:
                    countC += 1
                    feature.append(process[:-1])
                    label.append(process[-1])

    return feature,label

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


# 计算欧式距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))
    # 0ρ = sqrt( (x1-x2)^2+(y1-y2)^2 )　|x| = √( x2 + y2 )
    # power 对列表计算2次方  求和后开方

# 初始化质心随机样本
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape #获取数据集合的行列总数
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        # uniform() 方法将随机生成下一个实数，它在[x,y]范围内。
        centroids[i, :] = dataSet[index, :]
    return centroids


def computeRadius(indexInClust,dataSet,centerPoint):
    sum = 0.0
    maxR = 0.0
    for i in indexInClust:
        sum+=euclDistance(dataSet[i,:],centerPoint)
        #maxR = max(maxR,euclDistance(dataSet[i,:],centerPoint))

    return sum/size(indexInClust)
    #return maxR

# k-means cluster
def kmeansAlgorithm(dataSet, k):
    numSamples = dataSet.shape[0]#行数

    clusterAssment = mat(zeros((numSamples, 2))) #

    clusterChanged = True #停止循环标志位

    ## step 1: init 初始化k个质点
    centroids = initCentroids(dataSet, k)
    radiusCluster = [0] * k
    while clusterChanged:
        clusterChanged = False
        ## for each 行
        for i in xrange(numSamples):
            minDist  = 100000.0 # 设定一个极大值
            minIndex = 0
            ## for each centroid
            ## step 2: 寻找最接近的质心
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                # 将centroids（k个初始化质心）的j行和dataset（数据全集）的i行 算欧式距离，返回数值型距离
                if distance < minDist:
                # 找距离最近的质点，记录下来。
                    minDist  = distance
                    minIndex = j


            ## step 3: update its cluster # 跟新这个簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # clusterAssment 是一个n行2列的矩阵  Assment 评估
                clusterAssment[i, :] = minIndex, minDist**2 #赋值为 新的质点标号

        ## step 4: update centroids
        for j in range(k):
            # 属于j这个质点的所有数值的平均值算出成为新的质点
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis = 0)
            indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
            radiusCluster[j] = computeRadius(indexInClust, dataSet, centroids[j, :])

    print 'Congratulations, cluster complete!'
    return centroids, clusterAssment, radiusCluster

# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print "Sorry! Your k is too large! please contact Zouxy"
        return 1

    # 画出所有样例点 属于同一分类的绘制同样的颜色
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb'] #设定颜色

    # draw the centroids
    # 画出质点，用特殊图型
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

    plt.show()


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


def predict(dataSet, centroids, k, radiusCluster):
    m = dataSet.shape[0] #number of the dataSet
    count = 0.0
    outlierDist = []
    for i in range(m):
        #everyCluster = 0.0
        minDisClu = 100000000000000.0
        outlier = True
        for j in range(k):
            distance = euclDistance(dataSet[i,:],centroids[j,:])
            if distance<=radiusCluster[j]:
                outlier = False;
                break
            else:
                minDisClu = min(minDisClu, distance)
        #count+=1;

        if outlier == False:
            continue
        else:
            outlierDist.append(minDisClu)
            count+=1

    print "count",count
    print "m",m
    print "outlier accuracy: ", count/m
    print "outlier length", len(outlierDist)
    print "outlier mean", mean(outlierDist)
    print "outlier variance", var(outlierDist)


# if __name__ == '__main__':
#     ## step 1: 加载数据
#     print "step 1: load data..."
#     fileName = 'SynEDC';
#     sourcePath = 'C:/DataSet/Stream/Generate/dataset_set3_ori/'+str(fileName)+'/'+str(fileName)+'-bias_source_stream.csv'
#     targetPath = 'C:/DataSet/Stream/Generate/dataset_set3_ori/' + str(fileName) + '/' + str(
#         fileName) + '-bias_target_stream.csv'
#     sourceData, sourceLabel = load_data(sourcePath)
#     print "step 2: clustering..."
#
#     # mat 函数，将数组转化为矩阵
#
#     k = 5
#     centroids, clusterAssment,radiusCluster = kmeans(mat(sourceData), k)
#     purity = calcPurity(clusterAssment, k, sourceLabel)
#     print "purity", purity
#
#     #average
#     sumPurity = 0.0
#     for i in purity:
#         sumPurity+=i
#
#     print "avg purity",sumPurity/len(purity)
#
# testData,testLabel = readTestData(targetPath)
# print size(testLabel)
# testData = mat(testData)
# predict(testData,centroids,k,radiusCluster)


