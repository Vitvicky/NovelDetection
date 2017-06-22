# coding:UTF-8
'''
Date:20160923
@author: Zhuoyi Wang
'''

from numpy import *
import numpy as np
import random
from caculateFunction import load_data, ori_kmeans, distance, save_result
from kmeans import *

FLOAT_MAX = 1e100 # 设置一个较大的值作为初始化的最小的距离

def nearest(point, cluster_centers):
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in xrange(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

def get_centroids(points, k):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k , n)))
    # 1、随机选择一个样本点为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    # 2、初始化一个距离的序列
    d = [0.0 for _ in xrange(m)]

    for i in xrange(1, k):
        sum_all = 0
        for j in xrange(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random.random()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            break
    return cluster_centers

if __name__ == "__main__":
    k = 3
    sourcePath = 'C:/DataSet/Stream/CIKM_2017/original_data/fc/fc_0.1_source_stream.csv'
    #file_path = 'data/synthetic003-bias_source_stream.csv'
    print "---------- 1.load data ------------"
    data,label = load_data(sourcePath)
    print "---------- 2.K-Means++ generate centers ------------"
    centroids = get_centroids(mat(data), k)
    print "---------- 3.kmeans ------------"
    subCenter = ori_kmeans(mat(data), k, centroids)

    print len(subCenter)
    print "---------- 4.purity ------------"
    purity = calcPurity(subCenter, k, label)
    print "purity",purity