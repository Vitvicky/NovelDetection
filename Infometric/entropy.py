# -*- coding: utf-8 -*-
# !/usr/bin/env python

from numpy import *
from math import log

def entropy(label):
    length = len(label)
    #print "lengh of label: ",len(label)
    if length<1:
        return 0

    category = {}
    for i in label:
        if i not in category.keys():
            category[i] = 0

        category[i]+=1

    entropy = 0.0
    for key in category:
        prob = float(category[key])/length
        #print "prob",prob
        entropy -= prob * log(prob, 2)

    return entropy


    # noise = power(10, -10)
    # etpItem = label*log(label + noise)
    # etp = -sum(etpItem)
    #
    # return etp
# a = [0.00100071,0.00113857,0.00138947]
# print entropy(a)