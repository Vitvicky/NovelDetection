# -*- coding: utf-8 -*-
# !/usr/bin/env python

from Compute_Mutual_Info import *

def Object_Informetric(transferMatrix,sourceFeature,targetFeature,sourceLabel,lamda):

    #question, using mutual info to calculate source or target
    #do "discriminative clustering" on the target domain
    mi, grad = multual_info(transferMatrix,sourceFeature,sourceFeature,sourceLabel)

    #mutual info in both domains, make it hard to distinguish the two domains
    all_data = np.vstack((sourceFeature,targetFeature))
    all_data = np.array(all_data)
    #print "For all_data, it is a " + str(len(all_data)) + " * " + str(len(all_data[0])) + " 's marix"
    #all_label = [ones((len(sourceFeature), 1)),zeros((len(targetFeature), 1))]
    sourceLpart = [1]*len(sourceFeature)
    targetLpart = [0]*len(targetFeature)
    all_label = sourceLpart + targetLpart
    all_label = array(all_label)
    if lamda>0:
        mi2, grad2 = multual_info(transferMatrix, all_data, all_data, all_label)
        f = -(mi - dot(lamda,mi2))
        g = -(grad - dot(lamda,grad2))
    else:
        f = -mi
        g = -grad


    return f,g
