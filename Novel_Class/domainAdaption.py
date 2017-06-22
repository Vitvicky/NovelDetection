# -*- coding: utf-8 -*-
# !/usr/bin/env python


import sys
sys.path.append("..")
import scipy.io as sio
from sklearn.cluster import KMeans
from numpy import *
import numpy as np
from kmeans_lib.kmeans import kmeansAlgorithm,calcPurity,predict
from kmeans_lib.kmeans_kernel import KernelKMeans
from kmeans_lib.sklearn_kmeans_plus_plus import *

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
            if i<30000:
                i+=1
                continue;
            else:
                process = []
                line = rows.split(',')
                process = [float(x) for x in line]

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

def readTargetData(path):
    feature = []
    label = []
    count = 0
    with open(path) as file:
        for rows in file:
            if count<300:
                process = []
                line = rows.strip().split(',')
                #process = [float(x) for x in line]
                process = [float(x) for x in line]
                #print process[-1]
                if process[-1] == 4.0:
                #if line[-1] == 'vacuum_cleaning':
                    count+=1
                    feature.append(process[:-1])
                    label.append(process[-1])
    return feature,label



dataName = "Syndata_002"
rate = 0.2
d = 10
#sfPath = 'C:/Matlab_Code/infometric_0.1/Data/'+dataName+'/'+dataName+'_'+str(rate)+'.mat';
sfPath = 'C:/Matlab_Code/infometric_0.1/Data/'+dataName+'/'+dataName+'_'+str(rate)+'_d='+str(d)+'.mat';
#Lpath = 'C:/Matlab_Code/infometric_0.1/Data/L.mat';
targetPath = 'C:/Matlab_Code/infometric_0.1/Data/'+dataName+'/ori/'+dataName+'_'+str(rate)+'_target_stream.txt'

data = sio.loadmat(sfPath)
#Matrix = sio.loadmat(Lpath)

tranSourceFeature = data['tranSourceF']
sourceLabel = data['sourceLabel']
tranTargetFeature = data['trantargetF']
targetLabel = data['targetLabel']
transferMatrix = data['L']
print "sourceLabel ",sourceLabel
#original data
sourceFeature = data['sourceFeature']
targetFeature = data['targetFeature']

print "source data's length: ",len(tranSourceFeature)


#experiment 1
testData,testLabel = readTargetData(targetPath)
print "target outlier length: ",len(testLabel)
ori_testData = testData
K = 5
transferMatrix = np.array(transferMatrix).T
testData = np.array(testData).T
outlierTest = dot(transferMatrix,testData)
outlierTest = outlierTest.T
outlierTest.tolist()


#kmeans method
#centroids, clusterAssment,radiusCluster = kmeansAlgorithm(mat(tranSourceFeature), K)
centroids, clusterAssment,radiusCluster = keans_plus_plus(mat(tranSourceFeature), K,sourceLabel)


#kmeans kernel
#km = KernelKMeans(n_clusters=K, max_iter=100, random_state=0, verbose=1)
#predict_label =  km.fit_predict(mat(tranSourceFeature))
#
numSamples = len(data)
# clusterAssment = mat(zeros((numSamples, 1)))
# for i in xrange(numSamples):
#     clusterAssment[i, :] = predict_label[i]
purity = calcPurity(clusterAssment, K, sourceLabel)
print "purity", purity

sumPurity = 0.0
for i in purity:
    sumPurity+=i

print "avg purity",sumPurity/len(purity)
# kmeans = KMeans(n_clusters=K, random_state=0).fit(tranSourceFeature)
# kmeans.predict(tranSourceFeature)
# lenLabel = kmeans.labels_
# print lenLabel
# lenLabel = len(sk_predict)
#
# count = 0.0
# for j in range(lenLabel):
#     if sk_predict[j]+1 == targetLabel[j][0]:
#         count+=1
#
# print "target predict arruracy: ",count/len(targetLabel)
predict(mat(tranTargetFeature),centroids,K,radiusCluster)