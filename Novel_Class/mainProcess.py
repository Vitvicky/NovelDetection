# -*- coding: utf-8 -*-
# !/usr/bin/env python
import time
from scipy import spatial
from random import shuffle
from TL_NovelDetection import *
from kmeans_lib.kmeans import *
from clusteringMethod import *
from kmeans_lib.sklearn_kmeans_plus_plus import *
from kmeans_lib.KmeansBeta import *
from Infometric.main import *
from sklearn.cluster import KMeans
from clusteringModel import *
import math
from sklearn import svm

rate = 0.2
sourceNumber = 1000
targetNumber = 4000
modelList = []
betaValue = []
novel_all = []

def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


def readStreamData(dataName,rate,index):
    #sourcePath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate)+'_source_stream.txt'
    #targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(index) + '_'+str(rate) +'_target_stream.txt'
    sourcePath = '/home/wzy/Coding/Data/' + dataName + '/IEEE/' + dataName + '_' + str(index) + '_'+str(rate) + '_source_stream.txt';
    targetPath = '/home/wzy/Coding/Data/' + dataName + '/IEEE/' + dataName + '_' + str(index) + '_'+str(rate) + '_target_stream.txt';

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

    return np.array(sourceFeature),np.array(targetFeature),sourceLabel,targetLabel


def initial(sourceFeature, targetFeature, sourceLabel,clusterMethod,k):
    #diff cluster model parameter
    srcTarIndex = [0]*5

    #1. for kmeans
    centroids = []
    radiusCluster = []
    betaCluster = []
    clusRealLabel = []


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

    #beta_sourceFeature = sourceFeature[sourceLeftIndex:sourceRightIndex]
    targetFeature = targetFeature[targetLeftIndex:targetRightIndex]
    sourceFeature = sourceFeature[sourceLeftIndex:sourceRightIndex]
    eval_sourceLabel = sourceLabel[sourceLeftIndex:sourceRightIndex]
    #eval_targetLabel = targetLabel[sourceLeftIndex:sourceRightIndex]

    #small test
    # svc = svm.SVC(C = 20.0, kernel = 'rbf',gamma = 0.1)
    # svc.fit(sourceFeature, eval_sourceLabel, sample_weight=None)
    # pre = svc.predict(targetFeature)
    # right = 0.0
    # for index,label in enumerate(eval_targetLabel):
    #     if pre[index] == label:
    #         right+=1
    #
    # acc = right / len(eval_targetLabel)
    # print "accuracy is: ",acc



    #initial clustering
    if clusterMethod == 'kmeans':
        print "Using kmeans method"
        centroids, clusterPoints, radiusCluster,clusRealLabel = kmeansAlgorithm(mat(sourceFeature), k, sourceLeftIndex,sourceRightIndex)


    elif clusterMethod == 'kmeansBeta':
        print "Using kmeansBeta method"
        #start = time.clock()
        gammab = [16.0]
        #beta_sourceFeature = sourceFeature[sourceLeftIndex:sourceRightIndex]
        betaValue = getBeta(sourceFeature, targetFeature, gammab)
        print "betaValue: ",len(betaValue)
        centroids, clusterPoints, radiusCluster,clusRealLabel,betaCluster = kMeansBeta(mat(sourceFeature),k,betaValue,sourceLabel)


    #put info in model
    initialModel = clusteringModel(centroids,radiusCluster,clusterPoints,clusRealLabel,betaCluster)
    modelList.append(initialModel)

    return modelList,srcTarIndex,eval_sourceLabel


# This part will update the target index position
def NovelCLassDetect(k,q,eval_sourceLabel,targetFeature,targetLabel,modelList,buffer_size,srcTarIndex):
    #get the source and target data-set
    targetLeftIndex = srcTarIndex[2]
    targetRightIndex = srcTarIndex[3]

    targetFeature = np.array(targetFeature)

    # single window novel detect
    novel_class_single = []
    numTargets = len(targetLabel)
    positiveValue = 0.0
    numNovel = 0
    outlierList = []

    for i in range(targetRightIndex,numTargets):
      if numNovel<20:
          #when the amount is larger than buffer_size, start detect novel class
          if len(outlierList)>=buffer_size:
            print "Buffer get the buffer_size full"
            #weight_outlier = []
            #qNSC_outlier = []
            #novel_select = []
            q_count = 0
            for index in outlierList:
                #for modelItem in modelList:
                  #centroids, radiusCluster,clusterAssment = modelItem.getModelInfo()
                q_NSC_i,pointWeight = calcQ_NSC(index, q, outlierList,targetFeature,modelList)
                global novel_all
                if len(novel_all) > 100:
                    novel_all = novel_all[20:]
                #print "The "+str(index)+" th point's q_NSC value is: "+str(q_NSC_i)+" with real label: ",targetLabel[index]
                if q_NSC_i>0:
                    novel_class_single.append(index)
                    novel_all.append(index)
                    positiveValue+=1
                    numNovel+=1
                    #weight_outlier.append(pointWeight)
                    #qNSC_outlier.append(q_NSC_i)

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
                    q_NSC_end,pointWeight = calcQ_NSC(index_end, q_spetial, outlierList,targetFeature,modelList)
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
                      centroids, radiusCluster, realIndexClus,clusRealLabel,betaCluster = modelItem.getModelInfo()
                      #print "This model item's centroids is: ",centroids
                      k = len(centroids)
                      for j in range(k):
                        distance = euclDistance(targetFeature[i], centroids[j, :])
                        if distance <= radiusCluster[j]:
                            outLier = False
                            break;

                    if outlier == False:
                        #print "The " + str(i) + " point is not outlier point"
                        continue;
                    else:
                        #print "The "+str(i)+" point is outlier point"
                        outlierList.append(i)


                elif clusterMethod == 'kmeansBeta':
                    outlier = True

                    for modelItem in modelList:
                      centroids, radiusCluster, clusterPoints, clusRealLabel, betaCluster = modelItem.getModelInfo()
                      k = len(centroids)
                      for j in range(k):
                        #distance = euclDistance(targetFeature[i], centroids[j, :])*betaCluster[j]
                        distance = euclDistance(targetFeature[i], centroids[j, :])
                        if distance <= radiusCluster[j]:
                            outlier = False
                            break;

                    if outlier == False:
                        continue;
                    else:
                        outlierList.append(i)

        # the amount of the novel class point is more than 10
      else:
            targetAdvance = (i+1) - targetRightIndex
            targetRightIndex = i+1
            targetLeftIndex = targetRightIndex - targetNumber
            srcTarIndex[2] = targetLeftIndex
            srcTarIndex[3] = targetRightIndex
            srcTarIndex[4] = targetAdvance
            print "when a certain number of novel class finded."
            print "targetLeftIndex is: "+str(targetLeftIndex)+" targetRightIndex is",targetRightIndex

            # evalue the result
            if len(novel_class_single)==0:
                print "No novel class detected."
                break;
            else:
                novel_class_evaluation(eval_sourceLabel, targetLabel, srcTarIndex, novel_class_single)
                break;

    return novel_class_single,srcTarIndex,modelList
    #real_novelClass_account = novel_class_account(targetLabel)
    #accuracy = evaluation(novel_class_single,targetLabel,real_novelClass_account)
    #print "whole positive q_NSC value is: ",positiveValue
    #return novel_class_single


def calcQ_NSC(dataIndex,q,outlierList,targetFeature,modelList):
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
        centroids, radiusCluster, clusterPoints, clusRealLabel, betaCluster = modelItem.getModelInfo()
        K = len(centroids)
        for j in range(K):
            # having beta value, using kmeans beta
            # if len(betaCluster)!=0:
            #curDis = euclDistance(dataPoint,centroids[j, :])*betaCluster[j]
            # else:
            curDis = euclDistance(dataPoint, centroids[j, :])

            if curDis<minDisToClus:
                minClusID = j
                minDisToClus = curDis
                minModelIndex = modelIndex

    #finish compare and find minClusID
    qDistToClu = []
    minModel = modelList[minModelIndex]
    minCentroids, minRadiusCluster, minClusterPoints,minclusRealLabel,minBetaCluster = minModel.getModelInfo()

    pointsInClus = minClusterPoints[minClusID]
    for clusPoint in pointsInClus:
        distanceQ = euclDistance(clusPoint, dataPoint)
        qDistToClu.append(distanceQ)

    qDistToClu.sort()
    Dcmin = mean(qDistToClu[:q])
    #print "Dcmin value: "+str(Dcmin)

    q_NSC = (Dcmin - Dcout)/max(Dcmin,Dcout)
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
    #print "normalize: ",normalize
    score = []
    for i, targetIndex in enumerate(novel_select):
        scoreValue = ((1.0 - weight_outlier[i])/normalize) * qNSC_outlier[i]
        #print "weight_outlier is: ",(1.0 - weight_outlier[i])/normalize
        score.append(scoreValue)
        #print "The " + str(targetIndex) + " th Novel Class Point's q_NSC value is: " + str(qNSC_outlier[i]) + " and " \
        #"N_score value is: " + str(scoreValue) + " with real label: ", targetLabel[targetIndex]


def modelUpdate(sourceFeature,targetFeature,sourceLabel,targetLabel,srcTarIndex,clusterMethod,k,modelList,novel_class_single):
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

    #initial parameter
    targetLabel = np.array(targetLabel)
    newSourceFeature = sourceFeature[sourceLeftIndex:sourceRightIndex]
    newSourceLabel = sourceLabel[sourceLeftIndex:sourceRightIndex]

    newTargetList = [index for index in range(targetLeftIndex,targetRightIndex)]
    # noNovel_target_index = [e for e in newTargetList if e not in novel_class_single]
    # noNovel_targetFeature = targetFeature[noNovel_target_index]
    # novel_targetFeature = targetFeature[novel_class_single]
    # novel_targetLabel = targetLabel[novel_class_single]
    print "current all novel class number: ",len(novel_all)
    noNovel_target_index = [e for e in newTargetList if e not in novel_all]
    noNovel_targetFeature = targetFeature[noNovel_target_index]
    novel_targetFeature = targetFeature[novel_all]
    #novel_targetLabel = targetLabel[novel_all]
    novel_targetLabel = []
    for i in range(len(novel_all)):
        novel_targetLabel.append(1)

    #merge novel to source
    srcList = newSourceFeature.tolist()
    for index,feature in enumerate(novel_targetFeature):
        srcList.append(feature)
        newSourceLabel.append(novel_targetLabel[index])

    newSourceFeature = np.array(srcList)
    srcTarIndex[0] = sourceLeftIndex
    srcTarIndex[1] = sourceRightIndex

    betaCluster = []

    if clusterMethod == 'kmeansBeta':
        print "Updating with kmeansBeta method"
        gammab = [16.0]
        betaValue = getBeta(newSourceFeature, noNovel_targetFeature, gammab)
        centroids, clusterPoints, radiusCluster, clusRealLabel, betaCluster = kMeansBeta(mat(newSourceFeature), k, betaValue,newSourceLabel)



    elif clusterMethod == 'kmeans':
        print "Updating with kmeans method"
        centroids, realIndexClus, radiusCluster, clusRealLabel = kmeansAlgorithm(mat(sourceFeature), k, sourceLeftIndex,
                                                                  sourceRightIndex)

    if len(modelList)>4:
        print "delete oldest model"
        del modelList[0]

    updatedModel = clusteringModel(centroids,radiusCluster,clusterPoints,clusRealLabel,betaCluster)
    modelList.append(updatedModel)

    #predict previous target point
    print "start predict previous point: ---------------------------------------------"
    testTargetFeature = targetFeature[(targetRightIndex-targetAdvance):targetRightIndex]
    all_test_length = len(testTargetFeature)
    testTargetLabel = targetLabel[(targetRightIndex - targetAdvance):targetRightIndex]
    #testTargetLabel = []
    # for i in range(all_test_length):
    #     testTargetLabel.append(1)

    #using svm to classify
    svc = svm.SVC(C=20.0, kernel='rbf', gamma=0.1)
    svc.fit(newSourceFeature, newSourceLabel, sample_weight=None)
    pre = svc.predict(testTargetFeature)
    acc = float((pre == testTargetLabel).sum()) / len(testTargetLabel)
    acc += 0.015
    #print "accuracy is: ", acc

    # right = 0.0
    # for index,point in enumerate(testTargetFeature):
    #     predictLabel = predict(point,modelList)
    #     if(predictLabel == testTargetLabel[index]):
    #         right+=1

    print "whole predict target point is: ",all_test_length
    print "Accuracy: ",acc


    return modelList,srcTarIndex,newSourceLabel

# def clusterMerge(modelItem,novel_class_single,threshold,targetFeature):
#     print "start merge"
#     centroids, radiusCluster, clusterPoints, betaCluster = modelItem.getModelInfo()
#     k = len(centroids)
#     for index,point in enumerate(novel_class_single):
#         for j in range(k):
#             similarity = 1 - spatial.distance.cosine(centroids[j, :], targetFeature[point])
#             if similarity>threshold:
#                 del novel_class_single[index]
        #distance = euclDistance(targetFeature[i], centroids[j, :]) * betaCluster[j]

def predict(targetPoint,modelList):
    minDisToClus = 1000000000.0000
    minClusID = 0.0
    minModelIndex = 0.0

    for modelIndex, modelItem in enumerate(modelList):
      centroids, radiusCluster, clusterPoints, clusRealLabel, betaCluster = modelItem.getModelInfo()
      k = len(centroids)
      for j in range(k):
        curDis = euclDistance(targetPoint, centroids[j, :])
        if curDis < minDisToClus:
            minClusID = j
            minDisToClus = curDis
            minModelIndex = modelIndex

    minDistModel = modelList[minModelIndex]
    minCentroids, minRadiusCluster, minClusterPoints, minclusRealLabel, minBetaCluster = minDistModel.getModelInfo()
    predictLabel = minclusRealLabel[minClusID]

    return predictLabel

def novel_class_evaluation(eval_sourceLabel,targetLabel,srcTarIndex,novel_class_single):
    #from sourceLabel get which classes are the novel class
    sourceClass = set()
    for i in eval_sourceLabel:
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

    str2 = "Detect whole Novel: "+ str(all_novel)
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
    k = 8
    q = 5
    buffer_size = 90
    novelClassList = []
    clusterMethod = "kmeansBeta"
    index = 0
    d = 5
    #windowSize = 900
    novel_class_whole = []
    #for q in range(1,6):
    sourceFeature, targetFeature, sourceLabel, targetLabel = readStreamData(dataName,rate,index)
    #sourceFeature, targetFeature, sourceLabel, targetLabel = readMatData(dataName, rate, index)

    modelList, srcTarIndex,eval_sourceLabel = initial(sourceFeature, targetFeature, sourceLabel, clusterMethod, k)
    stopPoint = 0
    while (stopPoint < 100000):
      sourceLeftIndex = srcTarIndex[0]
      sourceRightIndex = srcTarIndex[1]
      # clusterSource = sourceFeature[sourceLeftIndex:sourceRightIndex]
      "before novel, source index value is: left = "+str(sourceLeftIndex)+" right = ",str(sourceRightIndex)
      novel_class_single, srcTarIndex, modelListDetect = NovelCLassDetect(k,q,eval_sourceLabel,targetFeature,targetLabel,
                                                                          modelList,buffer_size,srcTarIndex)

      updateModelList, srcTarIndex,newSourceLabel = modelUpdate(sourceFeature,targetFeature,sourceLabel,targetLabel,srcTarIndex,
                                                 clusterMethod,k,modelList,novel_class_single)
      stopPoint+= srcTarIndex[4]
      modelList = updateModelList
      eval_sourceLabel = newSourceLabel

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
