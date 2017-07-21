# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy
import numpy as np

# data = numpy.random.random(10)
# print "data: ",data
# bins = numpy.linspace(0, 1, 5)
# print "bins: ",bins
# digitized = numpy.digitize(data, bins)
# print "digitized: ",digitized
# #bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
# for i in range(1,len(bins)):
#
#     data_i = data[digitized == i]
#     print data_i


data = [0.87,0.92,0.77,0.56,0.48,0.58,0.62,0.81,0.49,0.35]
num_interval = 4
def calcCDF(ori_data,num_interval):
    length = len(ori_data)
    data = numpy.array(ori_data)
    pmf = [0]*num_interval
    cdf = [0]*num_interval
    bins = numpy.linspace(0, 1, num_interval+1)
    print "bins: ", bins
    digitized = numpy.digitize(data, bins)
    print "digitized: ", digitized
    sum = 0.0
    for i in range(1, len(bins)):
        data_i = data[digitized == i]
        print data_i
        pmf[i-1] = len(data_i)/(1.0*length)
        sum+= pmf[i-1]
        cdf[i-1] = sum

    return cdf

print calcCDF(data,num_interval)


def gini_coefficient(cdf,num_interval):
    print "this may not work"