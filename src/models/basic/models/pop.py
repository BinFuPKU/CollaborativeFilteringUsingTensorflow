
import numpy as np

import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV


class PopRank(object):
    def __init__(self, n_users, n_items,
                 topN=5, split_method='cv', eval_metrics=['rmse', 'mae']):
        self.__n_users, self.__n_items = n_users, n_items
        self.__topN, self.__split_method, self.__eval_metrics = topN, split_method, eval_metrics

        self.__pop = None
        self.__predict = None

    def __calpop__(self, trasR):
        pop_dict = {ind: trasR[:,ind].nnz for ind in range(trasR.shape[1])}
        return sorted(pop_dict.items(), key=lambda item:item[1], reverse=True)

    def __recommend(self, trasR, test_users):
        # tra
        tstintra_set = [set(trasR[user].nonzero()[1]) for user in test_users]

        yss_pred = []
        for ind in range(len(test_users)):
            yss_pred.append([])
            for tuple in self.__pop:
                if tuple[0] not in tstintra_set[ind]:
                    yss_pred[-1].append(tuple[0])
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
        self.__pop = self.__calpop__(trasR)

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





