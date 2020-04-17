
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sys
sys.path.append("./models/")
from svd import SVD
sys.path.append("../../utils/")
from IOUtil import loadSparseR
sys.path.append("../../samplers/")
from sampler_rating import Sampler

import multiprocessing

data_dir = '../../../data/'
cv_folds = 5
eval_metrics = ['rmse','mae', 'mse']

reg=.1

range_of_ratings=(1,5)
n_factors=32
batch_size=100


def worker(fold, n_users, n_items, dataset_dir):
    traFilePath = dataset_dir + 'ratings__' + str(fold + 1) + '_tra.txt'
    trasR = loadSparseR(n_users, n_items, traFilePath)

    print(dataset_dir.split('/')[-2]+':', trasR.shape, trasR.nnz, '%.2f' %(trasR.nnz/float(trasR.shape[0])))

    tra_tuple = np.array([(user, item, trasR[user, item]) for user, item in np.asarray(trasR.nonzero()).T]) # triad

    tstFilePath = dataset_dir + 'ratings__' + str(fold + 1) + '_tst.txt'
    tstsR = loadSparseR(n_users, n_items, tstFilePath)
    tst_tuple = np.array([(user, item, tstsR[user, item]) for user, item in np.asarray(tstsR.nonzero()).T]) # triad

    sampler = Sampler(trasR=trasR, negRatio=.0, batch_size=batch_size)
    svd = SVD(n_users, n_items, eval_metrics, range_of_ratings, reg, n_factors, batch_size)
    scores = svd.train(fold+1, tra_tuple, tst_tuple, sampler)

    print('fold=%d:' % fold, ','.join(['%s' % eval_metric for eval_metric in eval_metrics]), '=',
          ','.join(['%.6f' % (score) for score in scores]))

    return scores


if __name__ =='__main__':
    print('reg=',reg)

    # Movielens
    # ml-2k
    # dataset_dir = data_dir+'movielens/ml-2k/'
    # n_users, n_items = 2113, 10109
    # range_of_ratings=(.5,5)
    # ml-1m
    # dataset_dir = data_dir+'movielens/ml-1m/'
    # n_users, n_items = 6040, 3706
    # range_of_ratings=(1,5)
    # ml-100k
    dataset_dir = data_dir+'movielens/ml-100k/'
    n_users, n_items = 943, 1682

    # process pool
    folds_ = 1
    pool = multiprocessing.Pool(processes=folds_)
    results = []
    for cv_fold in range(folds_):
        results.append(pool.apply_async(func=worker, args=(cv_fold, n_users, n_items, dataset_dir)))
    pool.close()
    pool.join()
    pool.terminate()

    # scores = np.array([result.get() for result in results])
    # aves = scores.sum(0)/len(scores)
    # stds = np.sqrt(np.power(np.array(scores) - aves,2).sum(0)/len(scores))
    # print('ave=['+','.join(['%.4f' %ave for ave in aves])+']', 'std=['+','.join(['%.4f' %std for std in stds])+']')

