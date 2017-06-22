# -*- coding: utf-8 -*-
# !/usr/bin/env python

from numpy import *
from Object_Informetric import *
from kmeans_lib.kmeans import *

def load_data(path):

    feature = []
    label = []
    with open(path) as file:
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]

            feature.append(process[:-1])
            label.append(process[-1])

    return feature, label


def mainOperation(initialMatrix,sourceFeature,targetFeature,sourceLabel,lamda):
    L = initialMatrix
    d = len(L[0])
    MAXITER = 100
    stepsize = 1

    f = zeros((1, MAXITER))

    for iter in range(1,MAXITER):
        #print "+++++++++++++++++++++++++++++++++++++++"
        #print "The "+str(iter)+" th iteration: "
        #print "Start, L is a " + str(len(L)) + " * " + str(len(L[0])) + " 's matrix"
        f[0][iter], g = Object_Informetric(L, sourceFeature, targetFeature, sourceLabel, lamda)
        #print "In the "

        #adjust the stepsize adaptively
        if iter>1:
            if f[0][iter] < f[0][iter - 1]:
                stepsize = stepsize * 1.1;
            else:
                stepsize = stepsize * 0.5;

            if stepsize > 20:
                stepsize = 20

        # gradient descent
        a = stepsize*g
        L = L - stepsize*g
        #print "In GD, L is a " + str(len(L)) + " * " + str(L.shape[1]) + " 's matrix"

        #trace constraint
        ratio = d / trace(dot(L.T,L))
        L = dot(sqrt(ratio),L)

        #stopping condition
        if iter > 10 and max(abs(diff(f[0][iter - 3: iter] ) ) ) < 1e-6 * abs(f[0][iter]):
            break;

    return L

def clustering(sourceFeature,sourceLabel,targetFeature,K):
    #index = 1
    #type = "source"
    # sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_' + str(
    #     rate) + '_source_stream.txt';
    # sourcePath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_' + str(
    #     rate) + '_source_stream.txt';
    # targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_' + str(
    #     rate) + '_target_stream.txt';
    # targetPath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_' + str(
    #     rate) + '_target_stream.txt';

    # sourceFeature, sourceLabel = load_data(sourcePath)
    # targetFeature, targetLabel = load_data(targetPath)
    D = len(sourceFeature[0])
    d = 15
    initialMatrix = np.array(random.rand(D, d))
    lamda = 80
    sourceFeature = np.array(sourceFeature)
    #print "source data's length: ", len(sourceFeature)
    targetFeature = np.array(targetFeature)
    #print "target data's length: ", len(targetFeature)

    L = mainOperation(initialMatrix, sourceFeature, targetFeature, sourceLabel, lamda)
    tranSourceFeature = dot(sourceFeature, L).tolist()
    tranTargetFeature = dot(targetFeature, L).tolist()

    centroids, clusterAssment, radiusCluster = kmeansAlgorithm(mat(tranSourceFeature), K)

    return L, centroids, clusterAssment, radiusCluster