
import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.distance import  cosine

import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV


class UserItemCF(object):
    def __init__(self, n_users, n_items, topK=50, theta=.5,
                 topN=5, split_method='cv', eval_metrics=['rmse', 'mae']):
        self.__n_users, self.__n_items, self.__topK = n_users, n_items, topK
        self.__theta = theta
        self.__topN, self.__split_method, self.__eval_metrics = topN, split_method, eval_metrics

        self.__userSimMat = None
        self.__itemSimMat = None

    def __calsim__(self, sR):
        __simMat = (np.dot(sR, sR.T)).toarray()
        for ind in range(__simMat.shape[0]):
            den = np.linalg.norm(sR[ind,:].toarray())
            if den>0:
                __simMat[ind,:] = __simMat[ind,:] / den
                __simMat[:,ind] = __simMat[:,ind] / den
            __simMat[ind, ind] = 0
        return __simMat

    def __topk__(self, array):
        for ind in range(array.shape[0]):
            row_sim = np.zeros((array.shape[0]))
            for ind_ in np.argsort(array[ind, :])[-self.__topK:]:
                row_sim[ind_] = array[ind, ind_]
            array[ind, :] = row_sim
        return array

    def __predict__(self, trasR, test_users):
        predicts = []
        for user in test_users:
            user_predict = np.zeros(self.__n_items)
            for nn_user in np.argsort(self.__userSimMat[user,:])[-self.__topK:]:
                user_predict += self.__theta * self.__userSimMat[user,nn_user] * trasR[nn_user].toarray()[0]
            for item in trasR[user,:].nonzero()[1]:
                for item_ in self.__itemSimMat[item,:].nonzero()[0]:
                    user_predict[item_] += (1.-self.__theta) * self.__itemSimMat[item,item_] * trasR[user,item]
            predicts.append(user_predict)
        return predicts

    def __recommend(self, trasR, test_users):
        # tra
        tstintra_set = [set(trasR[user].nonzero()[1]) for user in test_users]
        itemset_maxsize = max([len(itemset) for itemset in tstintra_set])
        predicts = self.__predict__(trasR, test_users)

        yss_pred = []
        for ind in range(len(test_users)):
            yss_pred.append([])
            for item in np.argsort(predicts[ind])[-itemset_maxsize-self.__topN:][::-1]:
                if item not in tstintra_set[ind]:
                    yss_pred[-1].append(item)
                if len(yss_pred[-1]) >= self.__topN:
                    break
        return yss_pred

    def __eval(self, yss_true, yss_pred):
        if self.__split_method=='cv':
            return evaluateCV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        elif self.__split_method=='loov':
            return evaluateLOOV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        else:
            return None

    def train(self, fold, trasR, tstsR):
        # similarity matrix
        self.__userSimMat = self.__topk__(self.__calsim__(trasR))
        self.__itemSimMat = self.__topk__(self.__calsim__(trasR.T))

        # for eval:
        # tst
        test_users = list(set(np.asarray(tstsR.nonzero()[0])))
        yss_true = None
        if self.__split_method=='cv':
            yss_true = [set(tstsR[user].nonzero()[1]) for user in test_users]
        elif self.__split_method=='loov':
            yss_true = [tstsR[user].nonzero()[1][0] for user in test_users]

        yss_pred = self.__recommend(trasR, test_users)
        scores = self.__eval(yss_true, yss_pred)

        gc.collect()
        return scores





