# coding:UTF-8

import numpy as np

def load_data(path):
    feature = []
    label = []
    i = 0
    with open(path) as file:
        countA = 0
        countB = 0
        countC = 0
        countAll = 0
        for rows in file:
                process = []
                line = rows.split(',')
                process = [float(x) for x in line]
                # if process[-1] == 0.0 or process[-1] == 1.0 or process[-1] == 2.0:
                #     feature.append(process[:-1])
                #     label.append(process[-1])
                #     countAll+=1
                if process[-1] == 1.0 and countA!=300:
                    countA+=1
                    feature.append(process[:-1])
                    label.append(process[-1])

                elif process[-1] == 2.0 and countB!=300:
                    countB += 1
                    feature.append(process[:-1])
                    label.append(process[-1])
                elif process[-1] == 3.0 and countC != 300:
                    countC += 1
                    feature.append(process[:-1])
                    label.append(process[-1])

    return feature,label

def distance(vecA, vecB):
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]

def randCent(data, k):
    n = np.shape(data)[1]  # 属性的个数
    centroids = np.mat(np.zeros((k, n)))  # 初始化k个聚类中心
    for j in xrange(n):  # 初始化聚类中心每一维的坐标
        minJ = np.min(data[:, j])
        rangeJ = np.max(data[:, j]) - minJ
        # 在最大值和最小值之间随机初始化
        centroids[:, j] = minJ * np.mat(np.ones((k , 1))) + np.random.rand(k, 1) * rangeJ
    return centroids

def ori_kmeans(data, k, centroids):
    m, n = np.shape(data) # m：样本的个数，n：特征的维度
    subCenter = np.mat(np.zeros((m, 2)))  # 初始化每一个样本所属的类别
    change = True  # 判断是否需要重新计算聚类中心
    while change == True:
        change = False  # 重置
        for i in xrange(m):
            minDist = np.inf  # 设置样本与聚类中心之间的最小的距离，初始值为争取穷
            minIndex = 0  # 所属的类别
            for j in xrange(k):
                # 计算i和每个聚类中心之间的距离
                dist = distance(data[i, ], centroids[j, ])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            # 判断是否需要改变
            if subCenter[i, 0] <> minIndex:  # 需要改变
                change = True
                subCenter[i, ] = np.mat([minIndex, minDist])
        # 重新计算聚类中心
        for j in xrange(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0  # 每个类别中的样本的个数
            for i in xrange(m):
                if subCenter[i, 0] == j:  # 计算第j个类别
                    sum_all += data[i, ]
                    r += 1
            for z in xrange(n):
                try:
                    centroids[j, z] = sum_all[0, z] / r
                except:
                    print " r is zero"
    return subCenter

def save_result(file_name, source):
    m, n = np.shape(source)
    f = open(file_name, "w")
    for i in xrange(m):
        tmp = []
        for j in xrange(n):
            tmp.append(str(source[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()

if __name__ == "__main__":
    vecA = [1,2,3,4,5]
    vecB = [2,5,7,9,4]

    print distance(vecA,vecB)