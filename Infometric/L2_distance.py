# -*- coding: utf-8 -*-
# !/usr/bin/env python

from numpy import *
from numpy.matlib import *
import numpy as np

## matrixA - (DxM) matrix
## matrixB - (DxN) matrix

def L2_distance(listA,listB,df):
    matrixA = np.array(listA)
    matrixB = np.array(listB)
    #matrixA = listA
    #print "matrixA is a ", matrixA.shape
    #matrixB = listB
    #print "matrixB is a " ,matrixB.shape

    if (len(matrixA) == 1):
        matrixA = [matrixA,zeros((1, len(matrixA[0])))]
        matrixB = [matrixB,zeros((1, len(matrixB[0])))]


    AA = matrixA*matrixA;
    AA = AA.sum(axis=0)
    #print "AA is a "+str(len(AA)) +" 's matrix"
    BB = matrixB*matrixB;
    BB = BB.sum(axis=0)
    #print "BB is a " +str(len(BB)) + " 's matrix"
    AB = dot((matrixA.T),matrixB)
    #print "AB is a " + str(len(AB)) + "*" + str(AB.shape[1]) + " 's matrix"

    tempA = [0]*2
    tempA[0] = 1
    tempA[1] = len(BB)

    tempB = [0]*2
    tempB[0] = 1
    tempB[1] = len(AA)
    AA.reshape(len(AA),1)
    BB.reshape(len(BB),1)
    #print "AA.T",AA.T
    #d = repmat(AA.T,[1 size(bb,2)]) + repmat(BB,[size(AA,2) 1]) - 2*AB;
    rep1 = repmat(np.asmatrix(AA).T,1,len(BB))
    #print "rep1 is a " + str(len(rep1)) + "*" + str(rep1.shape[1]) + " 's matrix"
    rep2 = repmat(np.asmatrix(BB), len(AA),1)
    #print "rep2 is a " + str(len(rep2)) + "*" + str(rep2.shape[1]) + " 's matrix"
    d = rep1 + rep2 - 2*AB

    d = d.real
    if (df == 1):
        d = d* (1 - eye(size(d)));


    return d

