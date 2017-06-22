# -*- coding: utf-8 -*-
# !/usr/bin/env python
import sys
sys.path.append("..")
import scipy.io as sio
from sklearn.cluster import KMeans
from numpy import *
import numpy as np
from kmeans_lib.kmeans import *
from scipy.stats.stats import pearsonr
from sklearn.cluster import DBSCAN
from clusteringMethod import *

def readSourceData(dataName,d,rate,index):
    sfPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_d=' + str(
        d) + '.mat';

    data = sio.loadmat(sfPath)

    tranSourceFeature = data['tranSourceF']
    sourceLabel = data['sourceLabel']
    tranTargetFeature = data['trantargetF']
    targetLabel = data['targetLabel']
    sourceFeature = data['sourceFeature']
    targetFeature = data['targetFeature']

    return tranSourceFeature,sourceLabel,tranTargetFeature,targetLabel,sourceFeature,targetFeature

def readTargetData(dataName,rate):
    targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/ori/' + dataName + '_' + str(
        rate) + '_target_stream.txt'
    feature = []
    label = []
    countA = 0
    countB = 0
    countC = 0
    out = 0
    with open(targetPath) as file:
        for rows in file:
            process = []
            line = rows.strip().split(',')
            # process = [float(x) for x in line]
            process = [float(x) for x in line[:-1]]

            if process[-1] == 'other' and countA != 200:
                countA += 1
                feature.append(process)
                label.append(1)

            elif line[-1] == 'lying' and countB != 200:
                countB += 1
                feature.append(process)
                label.append(2)
            elif line[-1] == 'sitting' and countC != 200:
                countC += 1
                feature.append(process)
                label.append(3)

            if line[-1] == 'vacuum_cleaning' and out!=100:
                out += 1
                feature.append(process)
                label.append(4)

    return feature, label



def clusterInfo(clusterAssment, k, sourceLabel):
    purity = []
    for j in range(k):
        # everyInfo = 0.0
        maxPoint = 0.0
        minPoint = 0.0
        labelSet = {}
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]

        total = len(indexInClust) * 1.0
        #print "In" + str(j) + " th cluster, the total point is", total
        for i in indexInClust:
            pointLabel = sourceLabel[i][0]
            if pointLabel not in labelSet.keys():
                labelSet[pointLabel] = 0

            labelSet[pointLabel] += 1

        for key in labelSet:
            maxPoint = max(maxPoint, labelSet[key])

        print "maxLabel number", maxPoint
        purity.append(maxPoint / total)


    return max(purity),min(purity)

#########################################################
## parameter initial inluding pearsonr number
#########################################################
def parameterInitial(tranSourceFeature,k,sourceLabel):
    # kmeans method and initial correlation value install
    #modelInfo = []
    centroids, clusterAssment, radiusCluster = kmeansAlgorithm(mat(tranSourceFeature), k)
    maxPurity,minPurity = clusterInfo(clusterAssment, k, sourceLabel)
    correlation1 = []
    purityOfPoint = []
    predictOfPoint = []
    associateOfPoint = []
    numSource = len(tranSourceFeature)
    for i in xrange(numSource):
        # sourcePoint = tranSourceFeature[i]
        predictLabel, purity, minDisClu, assignClusIndex = nearsetClus(i, tranSourceFeature, sourceLabel, centroids, k,
                                                                       clusterAssment)
        association = radiusCluster[assignClusIndex] - minDisClu

        purityOfPoint.append(purity)
        associateOfPoint.append(association)
        if predictLabel == sourceLabel[i][0]:
            predictOfPoint.append(1)

        else:
            predictOfPoint.append(0)

    correlation1 = pearsonr(purityOfPoint, predictOfPoint)
    correlation2 = pearsonr(associateOfPoint, predictOfPoint)

    rP = correlation1/(correlation1+correlation2)
    rA = correlation2/(correlation1+correlation2)

    return centroids,clusterAssment,radiusCluster,rP,rA,maxPurity,minPurity


############################################################
## initial outlier detect just try different clustering method
############################################################
def initialOutlierDetect(tranSourceFeature,k,targetPoint,clusterMethod):
    outlier = True
    if clusterMethod == 'kmeans':
        print "Using kmeans method"
        centroids, clusterAssment, radiusCluster = kmeansAlgorithm(mat(tranSourceFeature), k)
        for j in range(k):
            distance = euclDistance(targetPoint, centroids[j, :])
            if distance <= radiusCluster[j]:
                outLier = False
                break;

        if outlier == False:
            return False;
        else:
            return True

    elif clusterMethod == 'dbscan':
        print "Using dbscan method"
        db_model = DBSCAN(eps=0.4, metric='euclidean', min_samples=10).fit(tranSourceFeature)
        labels = db_model.labels_
        #print labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        clusId2label,clusterAssment = clus_Info(labels, n_clusters, sourceLabel)

        return dbscan_predict(db_model, targetPoint, clusId2label)

    elif clusterMethod == 'BIRCH':
        print "Using dbscan method"

    elif clusterMethod == "kmeans++":
        print "Using dbscan method"

def normalizePurity(maxPurity,minPurity,targetPurity):
    if targetPurity<minPurity:
        minPurity = targetPurity

    return (minPurity - targetPurity)/(maxPurity - minPurity)


def novelDetect(dataName, rate, d, k, q):
    outlierList = []
    tranSourceFeature, sourceLabel, transferMatrix = readSourceData(dataName,d,rate)
    targetFeature, targetLabel = readTargetData(dataName,rate)
    numSource = len(tranSourceFeature)

    #transfer target data
    transferMatrix = np.array(transferMatrix).T
    targetT = np.array(targetFeature).T
    tranTargetFeature = dot(transferMatrix, targetT)
    tranTargetFeature = tranTargetFeature.T
    tranTargetFeature.tolist()


    numTargets = tranTargetFeature.shape[0]  # 行数
    print "numTargets: ",numTargets

    #initial cluster's information
    centroids, clusterAssment, radiusCluster, rP, rA, maxPurity, minPurity = parameterInitial(tranSourceFeature,k,sourceLabel)

    novel_class = []
    #novel detect
    for i in xrange(numTargets):
        dataPoint = tranTargetFeature[i, :]
        purity, association = confidenceScore(i, tranTargetFeature, clusterAssment, k, sourceLabel, centroids, radiusCluster)
        normalPurity = normalizePurity(maxPurity,minPurity,purity)
        confidence = rP*normalPurity + rA*association

        if len(outlierList)>=50:
            print "Buffer get the 50 point"
            #print "outlierList: ", outlierList
            #novel_item = []
            q_count = 0
            for index in outlierList:
                #print "outlierList real label is", targetLabel[index]
                q_NSC_i = calcQ_NSC(index, q, outlierList, K, clusterAssment, tranTargetFeature, tranSourceFeature)
                #print "outlier point's q_NSC value is: ", q_NSC_i
                if q_NSC_i>0:
                    q_count+=1
                    #novel_item.append(index)

            # finish statistic
            if q_count>q:
                print "novel class detect."
                novel_class.append(outlierList)
                #print outlierList
                outlierList = []
                print "================================================================================================"
            else:
                print "this is not a novel class."
                outlierList = []
                print "================================================================================================"

        else:
            # spetial situation, not reach 50 points, but reach the end of the dataset
            if i == numTargets-1:
                print "dataSet reach the end"
                lenOfList = len(outlierList)
                q_count_end = 0
                q_spetial = int(lenOfList*0.4)
                #print "q_spetial value: ",q_spetial
                for index_end in outlierList:
                    #print "outlierList real label is", targetLabel[index_end]
                    q_NSC_end = calcQ_NSC(index_end, q_spetial, outlierList, K, clusterAssment, tranTargetFeature, tranSourceFeature)
                    if q_NSC_end > 0:
                        q_count_end += 1
                if q_count_end > q_spetial:
                    print "novel class detect."
                    novel_class.append(outlierList)
                    outlierList = []
                else:
                    print "this is not a novel class."
                    outlierList = []

            else:
                if detect(dataPoint, centroids, K, radiusCluster) == False:
                    continue
                else:
                    #print "real label",targetLabel[i]
                    #outlierList.add(dataPoint)
                    outlierList.append(i)

    return novel_class

def calcQ_NSC(dataIndex,q,outlierList,K,clusterAssment,tranTargetFeature,tranSourceFeature):
    q_NSC = 0.0
    #print "dataIndex is: ",dataIndex
    dataPoint = tranTargetFeature[dataIndex,:]
    pointDistList = []
    for i in outlierList:
        if i!=dataIndex:
            otherPoint = tranTargetFeature[i,:]
            #print "the distance with outlier "+str(i)+" is: ",euclDistance(otherPoint,dataPoint)
            pointDistList.append(euclDistance(otherPoint,dataPoint))
        else:
            #print "i is"+str(i)+" dataIndex is: ",dataIndex
            continue

    pointDistList.sort()
    #calculate Dcout
    Dcout = mean(pointDistList[:q])
    print "Dcout: ",Dcout

    #calculate Dcmin
    Dcmin = 10000000000.0
    for j in range(K):
        qDistToClu = []
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
        for index in indexInClust:
            #print "index in cluster is: ",index
            qDistToClu.append(euclDistance(tranSourceFeature[index],dataPoint))

        qDistToClu.sort()
        meanValue = mean(qDistToClu[:q])
        #print "mean value of "+str(j)+" cluster is: "+str(meanValue)
        Dcmin = min(Dcmin,meanValue)
        print "Dcmin value: "+str(Dcmin)+" in the "+str(j)+" cluster."

    q_NSC = (Dcmin - Dcout)/max(Dcmin,Dcout)
    return q_NSC


def validationNovel(novelList,targetLabel):
    count = 0.0
    allPoint = len(novelList)
    for i in novelList:
        if targetLabel[i]!= 1 and targetLabel[i]!= 2 and targetLabel[i]!= 3:
            count+=1

    return count/allPoint

def confidenceScore(dataIndex,dataFeature,clusterAssment,k,dataLabel,centroids,radiusCluster):
    dataPoint = dataFeature[dataIndex]

    #find the closed cluster
    predictLabel, purity, minDisClu, assignClusIndex = nearsetClus(dataIndex, dataFeature, dataLabel, centroids, k, clusterAssment)
    # minDisClu = 100000000000000.0
    # nearCluIndex = 0
    # for j in range(K):
    #     indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
    #     distance = euclDistance(dataPoint, centroids[j, :])
    #     if distance < minDisClu:
    #         nearCluIndex = j
    #         minDisClu = distance
    #calculate the association
    association = radiusCluster[assignClusIndex] - minDisClu
    #confidence =


    return purity,association


# if __name__ == '__main__':
#     dataName = "pamap2"
#     rate = 0.1
#     d = 10
#     K = 5
#     q = 15
#     outlierList = []
#     tranSourceFeature, sourceLabel, transferMatrix = readSourceData(dataName,d,rate)
#     print "length of source", len(tranSourceFeature)
#     targetFeature, targetLabel = readTargetData(dataName,rate)
#     print "length of target", len(targetFeature)
#     outlierList = novelDetect(dataName, rate, d, transferMatrix, K, q)
#     print outlierList