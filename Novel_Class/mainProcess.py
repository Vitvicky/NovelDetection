# -*- coding: utf-8 -*-
# !/usr/bin/env python
import time
from novelDetection import *
from kmeans_lib.kmeans import *
from clusteringMethod import *
from kmeans_lib.sklearn_kmeans_plus_plus import *
from kmeans_lib.KmeansBeta import *
from Infometric.main import *

rate = 0.3
sourceNumber = 900
targetNumber = 2400

def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))



def readStreamData(dataName,d,rate,index):
    sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_source_stream.txt'
    targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) +'_target_stream.txt'
    # sourcePath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) + '_source_stream.txt';
    # targetPath = '/home/wzy/Coding/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) + '_target_stream.txt';

    sourceFeature = []
    targetFeature = []
    sourceLabel = []
    targetLabel = []

    with open(sourcePath) as file:
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]
            sourceFeature.append(process[:-1])
            sourceLabel.append(process[-1])

    with open(targetPath) as file:
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]
            targetFeature.append(process[:-1])
            targetLabel.append(process[-1])

    #print "after read, the length of source is: "+str(len(sourceFeature))+" and target length is: "+str(len(targetFeature))
    return np.array(sourceFeature),np.array(targetFeature),sourceLabel,targetLabel


#def initial(dataName, rate, index, k, q ,buffer_size,clusterMethod,windowSize,advance):
def initial(sourceFeature, targetFeature, sourceLabel, q, clusterMethod,k):
    #diff cluster model parameter
    d = 10
    srcTarIndex = [0]*4

    #1. for kmeans
    centroids = []
    clusterAssment = []
    radiusCluster = []
    betaValue = []

    #2. for dbscan
    db_model = 0
    clusId2label = {}

    outlierList = []

    L = []

    numSource = sourceNumber
    numTargets = targetNumber
    print "numSources in initial source: ", numSource
    print "numTargets in initial target: ",numTargets

    sourceLeftIndex = 0
    sourceRightIndex = sourceLeftIndex+numSource
    targetLeftIndex = 0
    targetRightIndex = targetLeftIndex+numTargets

    srcTarIndex[0] = sourceLeftIndex
    srcTarIndex[1] = sourceRightIndex
    srcTarIndex[2] = targetLeftIndex
    srcTarIndex[3] = targetRightIndex

    sourceFeature = sourceFeature[sourceLeftIndex:sourceRightIndex]
    targetFeature = targetFeature[targetLeftIndex:targetRightIndex]
    sourceLabel = sourceLabel[sourceLeftIndex:sourceRightIndex]

    #initial clustering
    if clusterMethod == 'kmeans':
        print "Using kmeans method"
        centroids, clusterAssment, radiusCluster = kmeansAlgorithm(mat(sourceFeature), k)
        #centroids, clusterAssment, radiusCluster = keans_plus_plus(mat(tranSourceFeature), K, sourceLabel)

    # elif clusterMethod == 'dbscan':
    #     db_model = DBSCAN(eps=0.4, metric='euclidean', min_samples=10).fit(sourceFeature)
    #     labels = db_model.labels_
    #     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #     clusId2label, clusterAssment = clus_Info(labels, n_clusters, sourceLabel)

    elif clusterMethod == 'kmeansBeta':
        print "Using kmeansBeta method"
        #start = time.clock()
        gammab = [0.001]
        betaValue = getBeta(sourceFeature, targetFeature, gammab)
        centroids, clusterAssment, radiusCluster = kMeansBeta(mat(sourceFeature), k,betaValue)
        #elapsed = (time.clock() - start)
        #print "time consuming is: ", elapsed

    elif  clusterMethod == 'domainAdapter':
        print "Using Domain Adapter method"
        L, centroids, clusterAssment, radiusCluster = clustering(sourceFeature,sourceLabel,targetFeature,k)
        print "centroids: ",centroids
        #sourceFeature = dot(sourceFeature, L).tolist()
        #targetFeature = np.array(dot(targetFeature, L).tolist())

    return clusterAssment,centroids,radiusCluster,L,srcTarIndex



def NovelCLassDetect(k,L,sourceFeature,targetFeature,targetLabel,clusterAssment,centroids,radiusCluster,buffer_size,srcTarIndex):
    #get the source and target data-set
    sourceLeftIndex = srcTarIndex[0]
    sourceRightIndex = srcTarIndex[1]
    targetLeftIndex = srcTarIndex[2]
    targetRightIndex = srcTarIndex[3]

    sourceFeature = np.array(dot(sourceFeature,L).tolist())
    targetFeature = np.array(dot(targetFeature,L).tolist())

    # novel detect
    novel_class = []
    novel_class_single = []
    numTargets = len(targetLabel)
    positiveValue = 0.0
    numNovel = 0
    outlierList = []

    for i in range(targetRightIndex,numTargets):
      dataPoint = targetFeature[i, :]
      #print "dataPoint: ",dataPoint
      if numNovel<10:
          #when the amount is larger than 50, start detect novel class
          if len(outlierList)>=buffer_size:
            print "Buffer get the 50 point"
            #print "outlierList: ", outlierList
            #novel_item = []
            q_count = 0
            for index in outlierList:
                q_NSC_i = calcQ_NSC(index, q, outlierList, K, clusterAssment, targetFeature, sourceFeature,centroids)
                print "The "+str(index)+" th point's q_NSC value is: "+str(q_NSC_i)+" with real label: ",targetLabel[index]
                if q_NSC_i>0:
                    positiveValue+=1
                    numNovel+=1
                    novel_class_single.append(index)
                    #novel_item.append(index)
            outlierList = []

          else:
            # spetial situation, not reach 50 points, but reach the end of the dataset
            if i == numTargets-1:
                print "dataSet reach the end"
                lenOfList = len(outlierList)
                q_count_end = 0
                q_spetial = int(lenOfList*0.3)
                #print "q_spetial value: ",q_spetial
                for index_end in outlierList:
                    #print "outlierList real label is", targetLabel[index_end]
                    q_NSC_end = calcQ_NSC(index_end, q_spetial, outlierList, K, clusterAssment, targetFeature, sourceFeature,
                                          centroids)
                    if q_NSC_end > 0:
                        numNovel += 1
                        positiveValue += 1
                        novel_class_single.append(index_end)

                outlierList = []


            #choose diff cluster method to detect outlier
            else:
                if clusterMethod == 'kmeans':
                    #print "choosing kmeans to detect outlier"
                    outlier = True
                    for j in range(k):
                        distance = euclDistance(targetFeature[i], centroids[j, :])
                        if distance <= radiusCluster[j]:
                            outLier = False
                            break;

                    if outlier == False:
                        continue;
                    else:
                        #print "The "+str(i)+" point is outlier point"
                        outlierList.append(i)

                # elif clusterMethod == 'dbscan':
                #     dbscanOutlier = dbscan_predict(db_model, targetFeature[i], clusId2label)
                #     if dbscanOutlier == False:
                #         continue;
                #     else:
                #         outlierList.append(i)

                elif clusterMethod == 'kmeansBeta'or clusterMethod == 'domainAdapter' :
                    outlier = True
                    for j in range(k):
                        distance = euclDistance(targetFeature[i], centroids[j, :])
                        if distance <= radiusCluster[j]:
                            outLier = False
                            break;

                    if outlier == False:
                        continue;
                    else:
                        # print "The "+str(i)+" point is outlier point"
                        outlierList.append(i)

                elif clusterMethod == 'domainAdapter':
                        outlier = True
                        for j in range(k):
                            distance = euclDistance(targetFeature[i], centroids[j, :])
                            if distance <= radiusCluster[j]:
                                outLier = False
                                break;

                        if outlier == False:
                            continue;
                        else:
                            # print "The "+str(i)+" point is outlier point"
                            outlierList.append(i)
        # the amount of the novel class point is more than 10
      else:
            targetLeftIndex = targetRightIndex
            targetRightIndex = i+1
            srcTarIndex[2] = targetLeftIndex
            srcTarIndex[3] = targetRightIndex
            break;

    return novel_class_single,srcTarIndex
    #real_novelClass_account = novel_class_account(targetLabel)
    #accuracy = evaluation(novel_class_single,targetLabel,real_novelClass_account)
    #print "whole positive q_NSC value is: ",positiveValue
    #return novel_class_single


def calcQ_NSC(dataIndex,q,outlierList,K,clusterAssment,targetFeature, sourceFeature,centroids):
    q_NSC = 0.0
    #print "dataIndex is: ",dataIndex
    dataPoint = targetFeature[dataIndex,:]
    pointDistList = []
    for i in outlierList:
        if i!=dataIndex:
            otherPoint = targetFeature[i,:]
            #print "the distance with outlier "+str(i)+" is: ",euclDistance(otherPoint,dataPoint)
            pointDistList.append(euclDistance(otherPoint,dataPoint))
        else:
            #print "i is"+str(i)+" dataIndex is: ",dataIndex
            continue

    pointDistList.sort()
    #calculate Dcout
    Dcout = mean(pointDistList[:q])
    #print "Dcout: ",Dcout



    #calculate Dcmin
    Dcmin = 10000000000.0

    #find the nearest cluster
    minDisToClus = 10000000000.0
    minClusID = 0
    for j in range(K):
        curDis = euclDistance(dataPoint,centroids[j, :])
        if curDis<minDisToClus:
            minClusID = j
            minDisToClus = curDis

    #finish compare and find minClusID
    qDistToClu = []
    indexInClust = nonzero(clusterAssment[:, 0].A == minClusID)[0]
    for index in indexInClust:
        qDistToClu.append(euclDistance(sourceFeature[index],dataPoint))

    qDistToClu.sort()
    Dcmin = mean(qDistToClu[:q])
    #print "Dcmin value: "+str(Dcmin)+" in the "+str(j)+" cluster."

    q_NSC = (Dcmin - Dcout)/max(Dcmin,Dcout)
    return q_NSC


def dominAdapterUpdate(sourceFeature,targetFeature,srcTarIndex,clusterMethod,k):
    #get the source,target window index
    sourceLeftIndex = srcTarIndex[0]
    sourceRightIndex = srcTarIndex[1]
    targetLeftIndex = srcTarIndex[2]
    targetRightIndex = srcTarIndex[3]

    targetAdvance = targetRightIndex - targetLeftIndex
    relation = (1 - rate) / rate
    sourceAdvance = int(targetAdvance / relation)
    targetAdvance = targetAdvance

    #update left,right index and get next data
    sourceLeftIndex = sourceRightIndex
    sourceRightIndex = sourceRightIndex+sourceAdvance
    print "sourceLeftIndex: ",sourceLeftIndex
    print "sourceRightIndex: ",sourceRightIndex

    newSourceFeature = sourceFeature[sourceLeftIndex:sourceRightIndex]
    newTargetFeature = targetFeature[targetLeftIndex:targetRightIndex]
    newSourceLabel = sourceLabel[sourceLeftIndex:sourceRightIndex]
    srcTarIndex[0] = sourceLeftIndex
    srcTarIndex[1] = sourceRightIndex


    if clusterMethod == 'domainAdapter':
        print "Updating with Domain Adapter method"
        D = len(newSourceFeature[0])
        d = 15
        initialMatrix = np.array(random.rand(D, d))
        lamda = 64
        newSourceFeature = np.array(newSourceFeature)
        newTargetFeature = np.array(newTargetFeature)

        # L = mainOperation(initialMatrix, sourceFeature, targetFeature, newSourceLabel, lamda)
        # tranSourceFeature = dot(sourceFeature, L).tolist()
        # tranTargetFeature = dot(targetFeature, L).tolist()
        #
        # centroids, clusterAssment, radiusCluster = kmeansAlgorithm(mat(tranSourceFeature), K)
        L, centroids, clusterAssment, radiusCluster = clustering(newSourceFeature, newSourceLabel, newTargetFeature, k)

    return L,centroids, clusterAssment, radiusCluster,srcTarIndex



def novel_class_account(targetLabel):
    account = 0.0
    for i in xrange(len(targetLabel)):
        if targetLabel[i][0] != 1.0 and targetLabel[i][0] != 6.0 and targetLabel[i][0] != 7.0:
            account+=1

    print "In target data, number of real novel class is: ",account
    return account




def evaluation(novel_class_single,targetLabel,real_novelClass_account):
    TruePredict = 0.0
    all_novel = 0.0
    # for outlier_list in novel_class:
    #     all_novel+=len(outlier_list)
    #     for outLier_id in outlier_list:
    #         if targetLabel[outLier_id][0] != 1.0 and targetLabel[outLier_id] != 6.0 and targetLabel[outLier_id] != 7.0:
    #             right_count+=1

    all_novel = len(novel_class_single)
    for outLier_id in novel_class_single:
            if targetLabel[outLier_id][0] != 1.0 and targetLabel[outLier_id][0] != 6.0 and targetLabel[outLier_id][0] != 7.0:
                TruePredict+=1

    FalsePredict = all_novel - TruePredict
    print "Whole target number is: ",len(targetLabel)
    print "Detect whole novel: ",all_novel
    print "Detect right novel: ",TruePredict
    print "Precision: ",TruePredict/(TruePredict+FalsePredict)
    print "Recall: ",TruePredict/real_novelClass_account


if __name__ == '__main__':
    dataName = "Syndata_002"
    #rate = 0.3
    K = 5
    q = 3
    buffer_size = 90
    novelClassList = []
    clusterMethod = "domainAdapter"
    index = 1
    d = 10
    windowSize = 900
    novel_class_whole = []
    #for q in range(1,6):
    sourceFeature, targetFeature, sourceLabel, targetLabel = readStreamData(dataName,d,rate,index)
    clusterAssment, centroids, radiusCluster, L, srcTarIndex = initial(sourceFeature, targetFeature, sourceLabel, q, clusterMethod,K)
    stopPoint = 0

    while (stopPoint < 500):
      sourceLeftIndex = srcTarIndex[0]
      sourceRightIndex = srcTarIndex[1]
      clusterSource = sourceFeature[sourceLeftIndex:sourceRightIndex]
      "before novel, source index value is: left = "+str(sourceLeftIndex)+" right = ",str(sourceRightIndex)
      novel_class_single, srcTarIndex = NovelCLassDetect(K,L,clusterSource,targetFeature,targetLabel,clusterAssment,centroids,radiusCluster,buffer_size,srcTarIndex)

      L, centroids, clusterAssment, radiusCluster, srcTarIndex = dominAdapterUpdate(sourceFeature,targetFeature,srcTarIndex,clusterMethod,K)
      stopPoint+= srcTarIndex[3] - srcTarIndex[2]


      print "============================================================================"
      print "novelClassList: ", novel_class_single
      novel_class_whole.append(novel_class_single)


    #sourceLabel, targetLabel, sourceFeature, targetFeature = readBetaSource(dataName, d, rate, index)
    #print "type: ",type(targetFeature)