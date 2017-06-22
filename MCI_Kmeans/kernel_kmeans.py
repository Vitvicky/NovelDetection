# -*- coding: utf-8 -*-
# !/usr/bin/env python
import sys
sys.path.append("..")
import scipy.io as sio
from sklearn.cluster import KMeans
from numpy import *
import numpy as np

from kmeans import kmeansAlgorithm,calcPurity,predict
from kernel_function import *
#from MCI-Kmeans import readTestData

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


dataName = "Syndata_c5"
rate = 0.5
d = 10
#sfPath = 'C:/Matlab_Code/infometric_0.1/Data/'+dataName+'/'+dataName+'_'+str(rate)+'.mat';
sfPath = 'C:/Matlab_Code/infometric_0.1/Data/'+dataName+'/'+dataName+'_'+str(rate)+'_d='+str(d)+'.mat';
#Lpath = 'C:/Matlab_Code/infometric_0.1/Data/L.mat';
targetPath = 'C:/Matlab_Code/infometric_0.1/Data/'+dataName+'/'+dataName+'_'+str(rate)+'_target_stream.txt'

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
testData,testLabel = readTestData(targetPath)
print "target outlier length: ",len(testLabel)
ori_testData = testData
K = 5
transferMatrix = np.array(transferMatrix).T
testData = np.array(testData).T
outlierTest = dot(transferMatrix,testData)
outlierTest = outlierTest.T
outlierTest.tolist()

sigma = 0.0005
kernelTranSourceFeature = kernel(tranSourceFeature, sigma)
centroids, clusterAssment,radiusCluster = kmeansAlgorithm(mat(kernelTranSourceFeature), K)
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
predict(mat(kernel(tranTargetFeature,sigma)),centroids,K,radiusCluster)