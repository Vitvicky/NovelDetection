# -*- coding: utf-8 -*-
# !/usr/bin/env python
import time
from random import shuffle
from mainProcess import *
from kmeans_lib.kmeans import *
from clusteringMethod import *
from Infometric.main import *
from clusteringModel import *
from transfer_learning.TCA import *
import math


rate = 0.2
sourceNumber = 500
targetNumber = 2000
modelList = []
betaValue = []
# tranSourceFeature = []
# tranTargetFeature = []


def initial(sourceFeature, targetFeature, sourceLabel, targetLabel,k):
    #diff cluster model parameter
    srcTarIndex = [0]*5

    centroids = []
    radiusCluster = []
    betaCluster = []
    betaSet = {}
    realIndexClus = {}

    dataInfo = []

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
    targetLabel = targetLabel[targetLeftIndex:targetRightIndex]
    #initial clustering

    print "Using TCA method"
    tca_model = TCA(dim=10, kerneltype='rbf', kernelparam=0.1, mu=1)
    source_tca, target_tca, x_tar_o_tca = tca_model.fit_transform(mat(sourceFeature), mat(targetFeature))
    centroids, realIndexClus, radiusCluster = kmeansAlgorithm(mat(source_tca), k)

    #put info into model and transfer data
    initialModel = clusteringModel(centroids,radiusCluster,realIndexClus,betaCluster,betaSet)
    dataInfo.append(source_tca)
    dataInfo.append(sourceLabel)
    dataInfo.append(target_tca)
    dataInfo.append(targetLabel)

    return srcTarIndex,initialModel,dataInfo

#detect the outlier point
def outlierDetect(k,dataIndex,model,tranTarFeature):
    centroids, radiusCluster, realIndexClus, betaCluster, betaSet = model.getModelInfo()
    dataPoint = tranTarFeature[dataIndex]
    outlier = True
    for j in range(k):
        distance = euclDistance(dataPoint, centroids[j,:])
        if distance <= radiusCluster[j]:
            outlier = False
            break;

    #finished iteration through all cluster
    if outlier == False:
        return False
    else:
        return True


# This part will update the target index position
def NovelCLassDetect(k,dataInfo,model,buffer_size):
    #get the feature and label of the src,tar
    tranSrcFeature = dataInfo[0]
    sourceLabel = dataInfo[1]
    tranTarFeature = dataInfo[2]
    targetLabel = dataInfo[3]

    sourceFeature = np.array(tranSrcFeature)
    targetFeature = np.array(tranTarFeature)


    # novel detect
    novel_class = []
    novel_class_single = []
    numTargets = len(targetLabel)
    positiveValue = 0.0

    outlierList = []

    for i in range(len(targetFeature)):
          #when the amount is larger than 50, start detect novel class
          if len(outlierList)>=buffer_size:
            print "Buffer get the define point"
            weight_outlier = []
            #qNSC_outlier = []
            #novel_select = []
            #q_count = 0
            for index in outlierList:
                q_NSC_i,pointWeight = calcQ_NSC(index, q, outlierList, targetFeature, sourceFeature,model)
                #print "The "+str(index)+" th point's q_NSC value is: "+str(q_NSC_i)+" with real label: ",targetLabel[index]
                if q_NSC_i>0:
                    novel_class_single.append(index)
                    positiveValue+=1
                    #novel_select.append(index)
                    #weight_outlier.append(pointWeight)
                    #qNSC_outlier.append(q_NSC_i)

            outlierList = []



          else:
            # spetial situation, not reach buffer size's points, but reach the end of the dataset
            if i == numTargets-1:
                print "dataSet reach the end"
                lenOfList = len(outlierList)
                q_count_end = 0
                q_spetial = int(lenOfList*0.3)
                #print "q_spetial value: ",q_spetial
                for index_end in outlierList:
                    q_NSC_end,pointWeight = calcQ_NSC(index_end, q_spetial, outlierList,targetFeature,sourceFeature,model)
                    if q_NSC_end > 0:
                        positiveValue += 1
                        novel_class_single.append(index_end)

                outlierList = []
                #N_score(novel_select, weight_outlier, qNSC_outlier, targetLabel)

            #choose diff cluster method to detect outlier
            else:
                judge = outlierDetect(k,i,model,targetFeature)
                if judge == True:
                    outlierList.append(i)
                else:
                    continue

    if len(novel_class_single) == 0:
        print "No novel class detected."
    else:
        novel_class_evaluation(sourceLabel, targetLabel, novel_class_single)

    return novel_class_single



def calcQ_NSC(dataIndex,q,outlierList,targetFeature,sourceFeature,model):
    q_NSC = 0.0
    dataPoint = targetFeature[dataIndex]
    pointDistList = []
    for i in outlierList:
        if i!=dataIndex:
            otherPoint = targetFeature[i]
            #print "the distance with outlier "+str(i)+" is: ",euclDistance(otherPoint,dataPoint)
            pointDistList.append(euclDistance(otherPoint,dataPoint))
        else:
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
    centroids, radiusCluster, realIndexClus, betaCluster, betaSet = model.getModelInfo()
    K = len(centroids)
    for j in range(K):
        curDis = euclDistance(dataPoint, centroids[j, :])

        if curDis<minDisToClus:
                minClusID = j
                minDisToClus = curDis


    #finish compare and find minClusID
    qDistToClu = []
    for key in realIndexClus:
        if realIndexClus[key] == minClusID:
            distanceQ = euclDistance(sourceFeature[key],dataPoint)
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


#update the index and model parameter
def modelUpdate(sourceFeature,targetFeature,sourceLabel,targetLabel,srcTarIndex,k):
    update_dataInfo = []
    #get the source,target window index
    sourceLeftIndex = srcTarIndex[0]
    sourceRightIndex = srcTarIndex[1]
    targetLeftIndex = srcTarIndex[2]
    targetRightIndex = srcTarIndex[3]

    #update left,right index and get next data
    sourceLeftIndex = sourceRightIndex
    sourceRightIndex = sourceRightIndex + sourceNumber
    targetLeftIndex = targetRightIndex
    targetRightIndex = targetRightIndex + targetNumber

    print "when update sourceLeftIndex: ",sourceLeftIndex
    print "when update sourceRightIndex: ",sourceRightIndex

    #initial parameter
    newSourceFeature = sourceFeature[sourceLeftIndex:sourceRightIndex]
    newTargetFeature = targetFeature[targetLeftIndex:targetRightIndex]
    newSourceLabel = sourceLabel[sourceLeftIndex:sourceRightIndex]
    newTargetLabel = targetLabel[targetLeftIndex:targetRightIndex]


    print "Updating with TCA method"
    betaCluster = []
    betaSet = {}
    newSourceFeature = np.array(newSourceFeature)
    tca_model = TCA(dim=10, kerneltype='rbf', kernelparam=0.1, mu=1)
    source_tca, target_tca, x_tar_o_tca = tca_model.fit_transform(mat(newSourceFeature), mat(newTargetFeature))
    centroids, realIndexClus, radiusCluster = kmeansAlgorithm(mat(source_tca), k)
    updateModel = clusteringModel(centroids, radiusCluster, realIndexClus, betaCluster, betaSet)

    update_dataInfo.append(source_tca)
    update_dataInfo.append(newSourceLabel)
    update_dataInfo.append(target_tca)
    update_dataInfo.append(newTargetLabel)


    return updateModel,srcTarIndex,update_dataInfo



def novel_class_evaluation(tran_sourceLabel,tran_targetLabel,novel_class_single):
    #from sourceLabel get which classes are the novel class
    sourceClass = set()
    for i in tran_sourceLabel:
        if i not in sourceClass:
            sourceClass.add(i)

    print "sourceClass is: ",sourceClass
    account = 0.0

    eval_targetLabel = tran_targetLabel
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

    index = 0
    #windowSize = 900
    novel_class_whole = []
    #for q in range(1,6):
    sourceFeature, targetFeature, sourceLabel, targetLabel = readStreamData(dataName,rate,index)
    #sourceFeature, targetFeature, sourceLabel, targetLabel = readMatData(dataName, rate, index)

    srcTarIndex, model, dataInfo = initial(sourceFeature, targetFeature, sourceLabel, targetLabel, k)
    iteration = 1
    while (iteration < 4):
      sourceLeftIndex = srcTarIndex[0]
      sourceRightIndex = srcTarIndex[1]
      # clusterSource = sourceFeature[sourceLeftIndex:sourceRightIndex]
      "before novel, source index value is: left = "+str(sourceLeftIndex)+" right = ",str(sourceRightIndex)
      novel_class_single = NovelCLassDetect(k,dataInfo,model,buffer_size)

      updateModel, update_srcTarIndex, update_dataInfo = modelUpdate(sourceFeature,targetFeature,sourceLabel,targetLabel,srcTarIndex,k)

      model = updateModel
      dataInfo = update_dataInfo
      srcTarIndex = update_srcTarIndex

      print "============================================================================"
      print "novelClassList: ", novel_class_single
      novel_class_whole.append(novel_class_single)
      iteration+=1

