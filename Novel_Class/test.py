# -*- coding: utf-8 -*-
# !/usr/bin/env python

import time
import sys
sys.path.append("..")
import scipy.io as sio
from sklearn.cluster import KMeans
from numpy import *
import numpy as np
from kmeans_lib.kmeans import kmeansAlgorithm,calcPurity,predict
from kmeans_lib.kmeans_kernel import KernelKMeans
from kmeans_lib.sklearn_kmeans_plus_plus import *
from Infometric.main import *

def getNovelClass(tranTargetFeature,targetLabel):
    real_outlier = []
    tranTarget = []
    for index, point in enumerate(targetLabel):
        if targetLabel[index] == 1.0 or targetLabel[index] == 6.0 or targetLabel[index] == 7.0:
            tranTarget.append(tranTargetFeature[index])
        else:
            real_outlier.append(tranTargetFeature[index])

    return real_outlier,tranTarget



if __name__ == '__main__':
    dataName = "Syndata_002"
    rate = 0.3
    d = 10

    index = 1
    type = "source"
    sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_' + str(
    rate) + '_source_stream.txt';
    targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_' + str(
    rate) +  '_target_stream.txt';


    sourceFeature,sourceLabel = load_data(sourcePath)
    targetFeature,targetLabel = load_data(targetPath)


    D = len(sourceFeature[0])
    d = 10
    # print "random no mat ",random.rand(D, d)
    #initialMatrix = random.rand(D,d)
    initialMatrix = np.array(random.rand(D, d))
    print "initialMatrix is a " + str(len(initialMatrix)) + "*" + str(initialMatrix.shape[1]) + " 's matrix"
    lamda = 64


    sourceFeature = np.array(sourceFeature)
    print "source data's length: ", len(sourceFeature)
    targetFeature = np.array(targetFeature)
    print "target data's length: ", len(targetFeature)
    #initialMatrix = np.mat(initialMatrix)
    #print initialMatrix
    start = time.clock()
    L = mainOperation(initialMatrix,sourceFeature,targetFeature,sourceLabel,lamda)
    print L.shape

    tranSourceFeature = dot(sourceFeature,L)
    tranTargetFeature = dot(targetFeature,L).tolist()
    #print "tranSourceFeature size: ", tranSourceFeature.shape
    #print "tranTargetFeature size: ", tranTargetFeature.shape

# #experiment 1
#
    K = 5
    #kmeans method
    centroids, clusterAssment,radiusCluster = kmeansAlgorithm(mat(tranSourceFeature), K)
    elapsed = (time.clock() - start)
    print "time consuming is: ",elapsed
    real_outlier,tranTarget = getNovelClass(tranTargetFeature,targetLabel)
    #tranTarget = mat(tranTarget)
    #print "tranTarget size is: ",tranTarget
    #centroids, clusterAssment,radiusCluster = keans_plus_plus(mat(tranSourceFeature), K,sourceLabel)
#
#
# #kmeans kernel
# #km = KernelKMeans(n_clusters=K, max_iter=100, random_state=0, verbose=1)
# #predict_label =  km.fit_predict(mat(tranSourceFeature))
# #
#
    purity = calcPurity(clusterAssment, K, sourceLabel)
    print "purity", purity

    sumPurity = 0.0
    for i in purity:
        sumPurity+=i

    print "avg purity",sumPurity/len(purity)

    predict(mat(real_outlier),centroids,K,radiusCluster)