# coding:UTF-8
import numpy as np
from numpy import *
import scipy as sp
import scipy.io as sio
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering

######################################################
#### necessray info for dbscan
######################################################

def clus_Info(labels,n_clusters,sourceLabel):
    numSamples = len(labels)
    clusterAssment = mat(zeros((numSamples, 1)))

    clusId2label = {}
    for i in xrange(numSamples):
        clusterAssment[i, :] = labels[i]
    #print "clusterAssment",clusterAssment
    for j in range(n_clusters):
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
        total = len(indexInClust) * 1.0
        maxPoint = 0.0
        labelSet = {}
        assignLabel = 0

        for item in indexInClust:
            pointLabel = sourceLabel[item][0]
            if pointLabel not in labelSet.keys():
                labelSet[pointLabel] = 0

            labelSet[pointLabel]+=1

        for key in labelSet:
            if labelSet[key] > maxPoint:
                maxPoint = labelSet[key]
                assignLabel = key

        purity = maxPoint/total
        print "In cluster id = "+str(j)+", assignLabel = "+str(assignLabel)+" purity = ",purity

        clusId2label[j] = assignLabel
        print"-----------------------------------------------------"

    return clusId2label,clusterAssment


def dbscan_predict(dbscan_model, X_new, clusId2label,metric=sp.spatial.distance.cosine):
    # Result is noise by default
    #y_new = np.ones(shape=len(X_new), dtype=int)*-1
    #y_label = np.ones(shape=len(X_new), dtype=int)*-1


    # Iterate all input samples for a label
    #for j, x_new in enumerate(X_new):
    outlier = True
        # Find a core sample closer than EPS
    for i, x_core in enumerate(dbscan_model.components_):
        if metric(X_new, x_core) <= dbscan_model.eps:
                outlier = False;
                break;

    if outlier == False:
        return False;
    else:
        return True;



#if __name__ == '__main__':
    # dataName = "pamap2"
    # rate = 0.3
    # d = 10
    #
    # sfPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(rate) + '_d=' + str(
    #     d) + '.mat';
    # targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/ori/' + dataName + '_' + str(
    #     rate) + '_target_stream.txt'
    #
    # data = sio.loadmat(sfPath)
    # # Matrix = sio.loadmat(Lpath)
    #
    # tranSourceFeature = data['tranSourceF']
    # sourceLabel = data['sourceLabel']
    # tranTargetFeature = data['trantargetF']
    # targetLabel = data['targetLabel']
    # transferMatrix = data['L']

    # testData, testLabel = readTargetData(targetPath)
    # #print "target outlier length: ", len(testLabel)
    # ori_testData = testData
    # transferMatrix = np.array(transferMatrix).T
    # testData = np.array(testData).T
    # outlierTest = dot(transferMatrix,testData)
    # outlierTest = outlierTest.T
    # outlierTest.tolist()


    #
    # #meanshift clustering algorithm
    # bandwidth = estimate_bandwidth(tranSourceFeature, quantile=0.3, n_samples=len(sourceLabel))
    # print "bandwidth: ",bandwidth
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms.fit(tranSourceFeature)
    # labels = ms.labels_
    # print "labels: ", labels
    # #cluster_centers = ms.cluster_centers_
    #
    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)
    #
    # print "number of estimated clusters : %d" % n_clusters_
    #
    # predict = ms.predict(tranTargetFeature)
    # clusId2label = clus_Info(labels, n_clusters_, sourceLabel)
    # #
    # print "====================================================================="
    #
    # count=0.0
    # predict_tran = np.ones(shape=len(predict), dtype=int) * -1
    # for index, predictId in enumerate(predict):
    #     predict_tran[index] = clusId2label[predictId]
    #     if predict_tran[index] == targetLabel[index]:
    #         count+=1
    #
    # print "len of target: ",len(targetLabel)
    # print count/len(targetLabel)


    #BIRCH algorithm
    # brc_model = Birch(branching_factor=40, n_clusters=None, threshold=0.4,compute_labels = True)
    # brc_model.fit(tranSourceFeature)
    # labels = brc_model.labels_
    # print "labels: ",labels
    # centroids = brc_model.subcluster_centers_
    # n_clusters = np.unique(labels).size
    # print "num of BIRCH cluster: ",n_clusters
    # clusId2label = clus_Info(labels, n_clusters, sourceLabel)
    #
    # print "center of clusters: ",centroids
    # #print "radius of cluster",radius(brc_model)
    #
    # predict_sameLabel = brc_model.predict(tranTargetFeature)
    #
    #
    # count_label = 0.0
    # predict_tran = np.ones(shape=len(predict_sameLabel), dtype=int) * -1
    # for index, predictId in enumerate(predict_sameLabel):
    #     predict_tran[index] = clusId2label[predictId]
    #     if predict_tran[index] == targetLabel[index]:
    #         count_label+=1
    #
    # print "len of same label target: ",len(targetLabel)
    # print count_label/len(targetLabel)


    #kmeans algorithm
    # kmeans = KMeans(n_clusters=8, random_state=0).fit(tranSourceFeature)
    # labels = kmeans.labels_
    # clusId2label = clus_Info(labels, 8, sourceLabel)
    # predict_same = kmeans.predict(tranTargetFeature)
    # count_label = 0.0
    # predict_tran = np.ones(shape=len(predict_same), dtype=int) * -1
    # for index, predictId in enumerate(predict_same):
    #     predict_tran[index] = clusId2label[predictId]
    #     if predict_tran[index] == targetLabel[index]:
    #         count_label+=1
    #
    # print "len of same label target: ",len(targetLabel)
    # print count_label/len(targetLabel)



