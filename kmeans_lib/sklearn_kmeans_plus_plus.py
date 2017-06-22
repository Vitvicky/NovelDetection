# -*- coding: utf-8 -*-
# !/usr/bin/env python

from numpy import *
import numpy as np
import random
from sklearn.cluster import KMeans
from Novel_Class.clusteringMethod import *
from kmeans import *

def keans_plus_plus(data,K,sourceLabel):
    radiusCluster = [0] * K
    kmeans_model = KMeans(n_clusters=K, random_state=0).fit(data)
    clustering_labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_
    clusId2label, clusterAssment = clus_Info(clustering_labels, K, sourceLabel)

    print "clusterAssment",clusterAssment
    for clus_index, clus_centor in enumerate(centroids):
    #for j in xrange(K):
        #pointsInCluster = data[nonzero(clusterAssment[:, 0].A == clus_index)[0]]
        indexInClust = nonzero(clusterAssment[:, 0].A == clus_index)[0]
        radiusCluster[clus_index] = computeSigmaRadius(indexInClust, data, clus_centor)

    return centroids, clusterAssment,radiusCluster
