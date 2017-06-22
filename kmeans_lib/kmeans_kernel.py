# coding:UTF-8

import numpy as np
from numpy import *
import scipy.io as sio
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

#from caculateFunction import *
from kmeans import *

class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """


    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in xrange(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print "Converged at iteration", it + 1
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in xrange(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)


if __name__ == '__main__':

    dataName = "Syndata_c5"
    rate = 0.5
    d = 10
    # sfPath = 'C:/Matlab_Code/infometric_0.1/Data/'+dataName+'/'+dataName+'_'+str(rate)+'.mat';
    sfPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(rate) + '_d=' + str(
        d) + '.mat';
    # Lpath = 'C:/Matlab_Code/infometric_0.1/Data/L.mat';
    targetPath = 'C:/Matlab_Code/infometric_0.1/Data/' + dataName + '/' + dataName + '_' + str(
        rate) + '_target_stream.txt'

    data = sio.loadmat(sfPath)
    # Matrix = sio.loadmat(Lpath)

    tranSourceFeature = data['tranSourceF']
    sourceLabel = data['sourceLabel']
    tranTargetFeature = data['trantargetF']
    targetLabel = data['targetLabel']
    transferMatrix = data['L']


    print "source data's length: ", len(tranSourceFeature)
    K = 3

    km = KernelKMeans(n_clusters=K, max_iter=100, random_state=0, verbose=1)
    cluster_label =  km.fit_predict(mat(tranSourceFeature))


    numSamples = len(tranSourceFeature)
    clusterAssment = mat(zeros((numSamples, 2)))
    for i in xrange(numSamples):
        clusterAssment[i, :] = cluster_label[i]
        #realLabel = sourceLabel[i][0]
        #print "real "+str(i)+" is: "+str(realLabel)+"..... actual is ",predict_label[i]

    purity = calcPurity(clusterAssment, K, sourceLabel)
    print "purity", purity
    predict_label = km.predict(mat(tranSourceFeature))
    count = 0.0
    for i in xrange(len(tranTargetFeature)):
        print "real " + str(i) + " is: " + str(targetLabel[i][0]) + "..... predict is ", predict_label[i]

        r_i = 0
        if targetLabel[i][0] == 3:
            r_i = 1
        elif targetLabel[i][0] == 1:
            r_i = 0
        else:
            r_i = 2


        if r_i == predict_label[i]:
            count+=1

    print count/len(targetLabel)





