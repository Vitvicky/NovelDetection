# -*- coding: utf-8 -*-
# !/usr/bin/env python
import time
import scipy.io as sio
from random import shuffle
from novelDetection import *
from kmeans_lib.kmeans import *
from clusteringMethod import *
from kmeans_lib.sklearn_kmeans_plus_plus import *
from kmeans_lib.KmeansBeta import *
from Infometric.main import *
from sklearn.cluster import KMeans
from clusteringModel import *
import math


rate = 0.2
sourceNumber = 1000
targetNumber = 4000
modelList = []

def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))

def readMatData(dataName,rate,index):
    sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index)+'_'+str(rate)  + '.mat';
    sourceLabel = []
    targetLabel = []
    data = sio.loadmat(sourcePath)
    tranSourceFeature = data['tranSourceF']
    sLabel = data['sourceLabel']
    tranTargetFeature = data['trantargetF']
    tLabel = data['targetLabel']
    transferMatrix = data['L']

    for i in sLabel:
        sourceLabel.append(i[0])
    for j in tLabel:
        targetLabel.append(j[0])

    return np.array(tranSourceFeature), np.array(tranTargetFeature), sourceLabel, targetLabel


def readStreamData(dataName,rate,index):
    #sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_source_stream.txt'
    #targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) +'_target_stream.txt'
    sourcePath = '/home/wzy/Coding/Data/' + dataName + '/ori/' + dataName + '_' + str(index) + '_'+str(rate) + '_source_stream.txt';
    targetPath = '/home/wzy/Coding/Data/' + dataName + '/ori/' + dataName + '_' + str(index) + '_'+str(rate) + '_target_stream.txt';

    sourceData = []
    sourceFeature = []
    targetFeature = []

    targetData = []
    sourceLabel = []
    targetLabel = []

    with open(sourcePath) as file:
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]
            sourceFeature.append(process[:-1])
            sourceLabel.append(process[-1])
            #sourceData.append(process)
    #shuffle(sourceData)

    with open(targetPath) as file:
        for rows in file:
            process = []
            line = rows.split(',')
            process = [float(x) for x in line]
            targetFeature.append(process[:-1])
            targetLabel.append(process[-1])
            #targetData.append(process)
    #shuffle(targetData)

    # for item in sourceData:
    #     sourceFeature.append(item[:-1])
    #     sourceLabel.append(item[-1])
    #     #print item[:-1]
    # for item1 in targetData:
    #     targetFeature.append(item1[:-1])
    #     targetLabel.append(item1[-1])

    return np.array(sourceFeature),np.array(targetFeature),sourceLabel,targetLabel


#def initial(dataName, rate, index, k, q ,buffer_size,clusterMethod,windowSize,advance):
def initial(sourceFeature, targetFeature, sourceLabel, q, clusterMethod,k):
    #diff cluster model parameter
    d = 10
    srcTarIndex = [0]*5

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
    srcTarIndex[4] = 0

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
        gammab = [10.0]
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
        return clusterAssment, centroids, radiusCluster, L, srcTarIndex

    #put info in model
    initialModel = clusteringModel(centroids,radiusCluster,clusterAssment)
    modelList.append(initialModel)

    return modelList,srcTarIndex


# This part will update the target index position
def NovelCLassDetect(k,L,sourceFeature,sourceLabel,targetFeature,targetLabel,modelList,buffer_size,srcTarIndex):
    #get the source and target data-set
    sourceLeftIndex = srcTarIndex[0]
    sourceRightIndex = srcTarIndex[1]
    targetLeftIndex = srcTarIndex[2]
    targetRightIndex = srcTarIndex[3]

    #sourceFeature = np.array(dot(sourceFeature,L).tolist())
    #targetFeature = np.array(dot(targetFeature,L).tolist())
    sourceFeature = np.array(sourceFeature)
    targetFeature = np.array(targetFeature)

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
            print "Buffer get the define point"
            weight_outlier = []
            qNSC_outlier = []
            novel_select = []
            q_count = 0
            for index in outlierList:
                #for modelItem in modelList:
                  #centroids, radiusCluster,clusterAssment = modelItem.getModelInfo()
                q_NSC_i,pointWeight = calcQ_NSC(index, q, outlierList, k, targetFeature, sourceFeature,modelList)
                #print "The "+str(index)+" th point's q_NSC value is: "+str(q_NSC_i)+" with real label: ",targetLabel[index]
                if q_NSC_i>0:
                    novel_class_single.append(index)
                    positiveValue+=1
                    numNovel+=1
                    novel_select.append(index)
                    weight_outlier.append(pointWeight)
                    qNSC_outlier.append(q_NSC_i)

            outlierList = []

            #calculate N_score
            #min_weight = min(weight_outlier)
            #normalize = 1 - min_weight
            #N_score(novel_select, weight_outlier, qNSC_outlier, targetLabel)


          else:
            # spetial situation, not reach buffer size's points, but reach the end of the dataset
            if i == numTargets-1:
                print "dataSet reach the end"
                novel_select = []
                weight_outlier = []
                qNSC_outlier = []
                lenOfList = len(outlierList)
                q_count_end = 0
                q_spetial = int(lenOfList*0.3)
                #print "q_spetial value: ",q_spetial
                for index_end in outlierList:
                    #print "outlierList real label is", targetLabel[index_end]
                    q_NSC_end,pointWeight = calcQ_NSC(index_end, q_spetial, outlierList,k,targetFeature,sourceFeature,modelList)
                    if q_NSC_end > 0:
                        numNovel += 1
                        positiveValue += 1
                        novel_class_single.append(index_end)
                        novel_select.append(index_end)
                        weight_outlier.append(pointWeight)
                        qNSC_outlier.append(q_NSC_end)

                outlierList = []
                #N_score(novel_select, weight_outlier, qNSC_outlier, targetLabel)

            #choose diff cluster method to detect outlier
            else:
                if clusterMethod == 'kmeans':
                    outlier = True
                    for modelItem in modelList:
                      centroids, radiusCluster, clusterAssment = modelItem.getModelInfo()
                      #print "This model item's centroids is: ",centroids
                      k = len(centroids)
                      for j in range(k):
                        distance = euclDistance(targetFeature[i], centroids[j, :])
                        if distance <= radiusCluster[j]:
                            outlier = False
                            break;

                    if outlier == False:
                        #print "The " + str(i) + " point is not outlier point"
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

                elif clusterMethod == 'kmeansBeta':
                    outlier = True

                    for modelItem in modelList:
                      centroids, radiusCluster, clusterAssment = modelItem.getModelInfo()
                      k = len(centroids)
                      for j in range(k):
                        distance = euclDistance(targetFeature[i], centroids[j, :])
                        if distance <= radiusCluster[j]:
                            outlier = False
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
                                outlier = False
                                break;

                        if outlier == False:
                            continue;
                        else:
                            # print "The "+str(i)+" point is outlier point"
                            outlierList.append(i)
        # the amount of the novel class point is more than 10
      else:
            targetAdvance = (i+1) - targetRightIndex
            targetRightIndex = i+1
            targetLeftIndex = targetRightIndex - targetNumber
            srcTarIndex[2] = targetLeftIndex
            srcTarIndex[3] = targetRightIndex
            srcTarIndex[4] = targetAdvance
            print "when 10 novel class find."
            print "targetLeftIndex is: "+str(targetLeftIndex)+" targetRightIndex is",targetRightIndex

            # evalue the result
            eva_sourceLabel = sourceLabel[sourceLeftIndex:sourceRightIndex]
            #eva_targetLabel = targetLabel[targetLeftIndex:targetRightIndex]
            if len(novel_class_single)==0:
                print "No novel class detected."
                break;
            else:
                novel_class_evaluation(eva_sourceLabel, targetLabel, srcTarIndex, novel_class_single)
                break;

    return novel_class_single,srcTarIndex,modelList
    #real_novelClass_account = novel_class_account(targetLabel)
    #accuracy = evaluation(novel_class_single,targetLabel,real_novelClass_account)
    #print "whole positive q_NSC value is: ",positiveValue
    #return novel_class_single


def calcQ_NSC(dataIndex,q,outlierList,K,targetFeature,sourceFeature,modelList):
    q_NSC = 0.0
    #print "dataIndex is: ",dataIndex
    dataPoint = targetFeature[dataIndex]
    pointDistList = []
    for i in outlierList:
        if i!=dataIndex:
            otherPoint = targetFeature[i]
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
    minModelIndex = 0
    for modelIndex,modelItem in enumerate(modelList):
        centroids, radiusCluster, clusterAssment = modelItem.getModelInfo()
        K = len(centroids)
        for j in range(K):
            curDis = euclDistance(dataPoint,centroids[j, :])
            if curDis<minDisToClus:
                minClusID = j
                minDisToClus = curDis
                minModelIndex = modelIndex

    #finish compare and find minClusID
    qDistToClu = []
    minModel = modelList[minModelIndex]
    minCentroids, minRadiusCluster, minClusterAssment = minModel.getModelInfo()
    indexInClust = nonzero(minClusterAssment[:, 0].A == minClusID)[0]
    for index in indexInClust:
        qDistToClu.append(euclDistance(sourceFeature[index],dataPoint))

    qDistToClu.sort()
    Dcmin = mean(qDistToClu[:q])
    #print "Dcmin value: "+str(Dcmin)+" in the "+str(j)+" cluster."

    q_NSC = (Dcmin - Dcout)/max(Dcmin,Dcout)
    if q_NSC>0:
        pointWeight = weight(dataPoint,minClusID,minCentroids,minRadiusCluster)
        #print "In outlier, the "+str(dataIndex)+" 's weight value is: ",pointWeight
    else:
        pointWeight = -1

    return q_NSC,pointWeight

def weight(dataPoint,minClusID,centroids, radiusCluster):
    r = radiusCluster[minClusID]
    d = euclDistance(dataPoint,centroids[minClusID, :])
    pointWeight = math.exp(r-d)

    return pointWeight

def N_score(novel_select,weight_outlier, qNSC_outlier,targetLabel):
    min_weight = min(weight_outlier)
    normalize = 1.0 - min_weight
    print "normalize: ",normalize
    score = []
    for i, targetIndex in enumerate(novel_select):
        scoreValue = ((1.0 - weight_outlier[i])/normalize) * qNSC_outlier[i]
        #print "weight_outlier is: ",(1.0 - weight_outlier[i])/normalize
        score.append(scoreValue)
        #print "The " + str(targetIndex) + " th Novel Class Point's q_NSC value is: " + str(qNSC_outlier[i]) + " and " \
        #"N_score value is: " + str(scoreValue) + " with real label: ", targetLabel[targetIndex]


def modelUpdate(sourceFeature,targetFeature,srcTarIndex,clusterMethod,k,modelList):
    #get the source,target window index
    sourceLeftIndex = srcTarIndex[0]
    sourceRightIndex = srcTarIndex[1]
    targetLeftIndex = srcTarIndex[2]
    targetRightIndex = srcTarIndex[3]



    #calculating the changed index
    targetAdvance = srcTarIndex[4]
    relation = (1 - rate) / rate
    sourceAdvance = int(targetAdvance / relation)
    #targetAdvance = targetAdvance

    #update left,right index and get next data
    sourceRightIndex = sourceRightIndex + sourceAdvance
    sourceLeftIndex = sourceRightIndex - sourceNumber

    print "when update sourceLeftIndex: ",sourceLeftIndex
    print "when update sourceRightIndex: ",sourceRightIndex

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
        newTargetFeature = np.array(newSourceFeature)
        # centroids, clusterAssment, radiusCluster = kmeansAlgorithm(mat(tranSourceFeature), K)
        L, centroids, clusterAssment, radiusCluster = clustering(newSourceFeature, newSourceLabel, newTargetFeature, k)
        return L, centroids, clusterAssment, radiusCluster, srcTarIndex

    elif clusterMethod == 'kmeansBeta':
        print "Updating with kmeansBeta method"
        gammab = [10.0]
        betaValue = getBeta(newSourceFeature, newTargetFeature, gammab)
        centroids, clusterAssment, radiusCluster = kMeansBeta(mat(newSourceFeature), k, betaValue)



    elif clusterMethod == 'kmeans':
        print "Updating with kmeans method"
        #betaValue = getBeta(newSourceFeature, newTargetFeature, gammab)
        centroids, clusterAssment, radiusCluster = kmeansAlgorithm(mat(newSourceFeature), k)

    if len(modelList)>4:
        print "delete oldest model"
        del modelList[0]

    updatedModel = clusteringModel(centroids, radiusCluster, clusterAssment)
    modelList.append(updatedModel)

    return modelList,srcTarIndex



def novel_class_evaluation(sourceLabel,targetLabel,srcTarIndex,novel_class_single):
    #from sourceLabel get which classes are the novel class
    sourceClass = set()
    for i in sourceLabel:
        if i not in sourceClass:
            sourceClass.add(i)

    print "sourceClass is: ",sourceClass
    account = 0.0

    targetAdvanced = srcTarIndex[4]
    targetRightIndex = srcTarIndex[3]
    eval_targetLabel = targetLabel[targetRightIndex - targetAdvanced:targetRightIndex]
    targetClass = set()
    for i in eval_targetLabel:
        if i not in targetClass:
            targetClass.add(i)

    print "targetClass is: ", targetClass

    novel_class = set()
    for i in xrange(len(eval_targetLabel)):
        # if targetLabel[i] != 1.0 and targetLabel[i] != 6.0 and targetLabel[i] != 7.0:
        if eval_targetLabel[i] not in sourceClass:
            novel_class.add(eval_targetLabel[i])
            account += 1

    print "In target data, number of real novel class is: ", account
    print "novel class set", novel_class
    #return account,sourceClass

    #evaluating the detect result
    TruePredict = 0.0
    all_novel = len(novel_class_single)
    for outLier_id in novel_class_single:
        #if targetLabel[outLier_id] != 1.0 and targetLabel[outLier_id] != 6.0 and targetLabel[outLier_id] != 7.0:
        if targetLabel[outLier_id] not in sourceClass:
            TruePredict += 1

    FalsePredict = all_novel - TruePredict

    writePath = "detect_evaluation.txt"
    file = open(writePath, 'w')
    str1 = "Whole target number is: "+str(len(eval_targetLabel))
    print str1
    file.write(str1);
    file.write("\n")

    str2 = "Detect whole outlier: "+ str(all_novel)
    print str2
    file.write(str2);
    file.write("\n")

    str3 = "Detect right novel: "+ str(TruePredict)
    print str3
    file.write(str3);
    file.write("\n")

    if account == 0:
        str4 = "No novel class in the targetData, Precision is "+str(TruePredict / (TruePredict + FalsePredict))
    else:
        str4 = "Precision: "+ str(TruePredict / (TruePredict + FalsePredict))+ " Recall: "+str(TruePredict / account)
    print str4
    file.write(str4);
    file.write("\n")


if __name__ == '__main__':
    dataName = "Syndata_002"
    #rate = 0.3
    k = 5
    q = 5
    buffer_size = 100
    novelClassList = []
    clusterMethod = "kmeansBeta"
    index = 0
    d = 10
    #windowSize = 900
    novel_class_whole = []
    #for q in range(1,6):
    sourceFeature, targetFeature, sourceLabel, targetLabel = readStreamData(dataName,rate,index)
    #sourceFeature, targetFeature, sourceLabel, targetLabel = readMatData(dataName, rate, index)

    modelList, srcTarIndex = initial(sourceFeature, targetFeature, sourceLabel, q, clusterMethod, k)
    stopPoint = 0
    L = []
    while (stopPoint < 1000):
      sourceLeftIndex = srcTarIndex[0]
      sourceRightIndex = srcTarIndex[1]
      clusterSource = sourceFeature[sourceLeftIndex:sourceRightIndex]
      "before novel, source index value is: left = "+str(sourceLeftIndex)+" right = ",str(sourceRightIndex)
      novel_class_single, srcTarIndex, modelListDetect = NovelCLassDetect(k,L,clusterSource,sourceLabel,targetFeature,targetLabel,modelList,
                                                                      buffer_size,srcTarIndex)

      updateModelList, srcTarIndex = modelUpdate(sourceFeature,targetFeature,srcTarIndex,clusterMethod,k,modelListDetect)
      stopPoint+= srcTarIndex[4]
      modelList = updateModelList

      print "============================================================================"
      print "novelClassList: ", novel_class_single
      novel_class_whole.append(novel_class_single)

    #evaluate label
    # targetStop = targetLabel[targetNumber:targetNumber+stopPoint]
    # real_novelClass_account = novel_class_account(targetStop)
    #
    # novel_class_set = []
    # for single_set in novel_class_whole:
    #     for novel_item in single_set:
    #         novel_class_set.append(novel_item)
    #
    # evaluation(novel_class_set, targetLabel, real_novelClass_account)

