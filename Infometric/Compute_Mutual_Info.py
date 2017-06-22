# -*- coding: utf-8 -*-
# !/usr/bin/env python

from numpy import *
from L2_distance import *
from entropy import *
from sklearn.metrics.pairwise import euclidean_distances

def multual_info(transferMatrix,sourceFeature,targetFeature,sourceLabel):
    noise = pow(10, -10)
    nSource = len(sourceFeature)
    nTarget = len(targetFeature)
    dimension = len(sourceFeature[0])
    #print "For transferMatrix, it is a " , transferMatrix.shape
    #print "For sourceFeature, it is a " , sourceFeature.shape
    #print "For targetFeature, it is a " , targetFeature.shape
    Dist = L2_distance(dot(transferMatrix.T,sourceFeature.T), dot(transferMatrix.T,targetFeature.T),0)
    #Dist = euclidean_distances(dot(transferMatrix.T, sourceFeature.T), dot(transferMatrix.T, targetFeature.T))
    #print "For Dist, it is a "+str(len(Dist))+" * "+str(Dist.shape[1])+" 's marix"

    expD = exp(-Dist)
    sumItem = expD.sum(axis=0)
    P = expD/repmat(np.asmatrix(sumItem), nSource, 1)
    #print "P is a " + str(len(P)) + "*" + str(P.shape[1]) + " 's matrix"
    #P = expD / np.tile(sum(expD), (nSource, 1))

    P = P - diag(diag(P));
    N_Class = len(unique(sourceLabel))
    #print "N_class is: ",N_Class

    P_c = zeros((N_Class, nTarget))
    Idx = zeros((nSource, N_Class))
    classHash,classAssment = labelStatis(sourceLabel,N_Class)

    for i in range(0,N_Class):
        realLabel = classHash[i]
        #id = (sourceLabel == i+1);
        idSet = nonzero(classAssment[:, 0].A == realLabel)[0]
        #for id in idSet:
        Idx[idSet, i] = 1;
        R = expD[idSet,:].sum(axis=0)

        P_c[i,:] = P[idSet,:].sum(axis=0)


    val1 = P_c.sum(axis=1) / nTarget
    #print "P_c.flatten: ",P_c.flatten()
    mi = entropy(val1)
    mi = mi - entropy(P_c.flatten()) / nTarget
    #compute gradient
    temp = np.log(val1 + noise)
    temp = np.asmatrix(temp)
    alpha_ct = (np.log(P_c + noise) - repmat(temp.T, 1, nTarget)) / nTarget
    #print "alpha_ct size is: " + str(len(alpha_ct)) + " *　" + str((alpha_ct).shape[1])

    alpha = zeros((nSource, nTarget))
    for i in range(0,N_Class):
        realLabel = classHash[i]

        idSet = nonzero(classAssment[:, 0].A == realLabel)[0]
        #print "Belong to label: "+str(realLabel)+", the point is: ",idSet
        rep = repmat(np.asmatrix(alpha_ct[i, :]), len(idSet), 1)
        alpha[idSet, :] = rep
    #print "alpha size is: " + str(len(alpha)) + " *　" + str(len(alpha[0])) + " matrix"

    #gamma
    sumValue = sum(alpha*P)
    gamma = (repmat(np.asmatrix(sumValue), nSource, 1) - alpha)* P;
    g = -dot(dot(sourceFeature.T, gamma),targetFeature) - dot(dot(targetFeature.T, gamma.T),sourceFeature)
    grad = dot(dot(2,g),transferMatrix)
    #print "grad: ",grad
    # etp = entropy(val)
    #
    # noise = power(10, -10)
    # etpItem = val*log(val + noise)
    # etp = -etpItem.sum(axis=0)

    return mi,grad


def labelStatis(label,N_Class):
    classHash = [0]*N_Class
    classAssment = mat(zeros((len(label), 1)))
    category = {}
    for index,item in enumerate(label):
        classAssment[index, :] = label[index]
        if item not in category.keys():
            category[item] = 0

        category[item]+=1

    #dict(zip())
    count=0
    for key in category:
        if count<N_Class:
            classHash[count] = key
            count+=1

    return classHash,classAssment