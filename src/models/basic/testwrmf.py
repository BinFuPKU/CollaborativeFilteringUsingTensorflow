
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sys
sys.path.append("./models/")
from wrmf import WRMF
sys.path.append("../../utils/")
from IOUtil import loadSparseR
from Util import matBinarize
sys.path.append("../../samplers/")
from sampler_rating import Sampler

import multiprocessing
from scipy.sparse import lil_matrix

data_dir = '../../../data/'
folds = 5
binarize_threshold = 3

weight=2.
reg=.1

topN=10
split_method='cv'
eval_metrics = ['pre','recall', 'map', 'mrr', 'ndcg']
negRatio=1
n_factors=100
batch_size=100


def worker(fold, n_users, n_items, dataset_dir):
    traFilePath = dataset_dir + 'ratings__' + str(fold + 1) + '_tra.txt'
    trasR = lil_matrix(matBinarize(loadSparseR(n_users, n_items, traFilePath), binarize_threshold))

    print(dataset_dir.split('/')[-2] + '@%d:' % (fold + 1), trasR.shape, trasR.nnz,
          '%.2f' % (trasR.nnz / float(trasR.shape[0])))

    tstFilePath = dataset_dir + 'ratings__' + str(fold + 1) + '_tst.txt'
    tstsR = lil_matrix(matBinarize(loadSparseR(n_users, n_items, tstFilePath), binarize_threshold))

    sampler = Sampler(trasR, negRatio, batch_size)
    wrmf = WRMF(n_users, n_items, topN, split_method, eval_metrics, weight, reg, n_factors, batch_size)
    scores = wrmf.train(fold+1, trasR, tstsR, sampler)

    print(dataset_dir.split('/')[-2] + '@%d:' % (fold + 1),'weight=',weight,'reg=',reg)
    print('fold=%d:' % (fold+1), ','.join(['%s' % eval_metric for eval_metric in eval_metrics]), '=',
          ','.join(['%.6f' % (score) for score in scores]))

    return scores


if __name__ =='__main__':
    print('weight=',weight,'reg=',reg)

    # Movielens
    # ml-100k
    dataset_dir = data_dir+'movielens/ml-100k/'
    n_users, n_items = 943, 1682
    # ml-2k
    # dataset_dir = data_dir+'movielens/ml-2k/'
    # n_users, n_items = 2113, 10109
    # ml-1m
    # dataset_dir = data_dir+'movielens/ml-1m/'
    # n_users, n_items = 6040, 3706

    # lastfm2k
    # dataset_dir = data_dir+'lastfm/lastfm2k/'
    # n_users, n_items = 1892, 17632

    # lastfm2k
    # dataset_dir = data_dir+'netflix5k5k/'
    # n_users, n_items = 4744, 3550

    # Amazon
    # dataset_dir = data_dir+'Amazon/'
    # n_users, n_items = 6170, 2753

    # douban Movie_
    # dataset_dir = data_dir+'douban/Movie_/'
    # n_users, n_items = 13367, 12677

    # process pool
    pool = multiprocessing.Pool(processes=3)
    results = []
    for fold in range(3):
        results.append(pool.apply_async(func=worker, args=(fold, n_users, n_items, dataset_dir)))
    pool.close()
    pool.join()
    pool.terminate()

    scores = np.array([result.get() for result in results])
    aves = scores.sum(0)/len(scores)
    stds = np.sqrt(np.power(np.array(scores) - aves,2).sum(0)/len(scores))
    print('ave@'+str(topN)+'=['+','.join(['%.4f' %ave for ave in aves])+']', 'std@'+str(topN)+'=['+','.join(['%.4f' %std for std in stds])+']')
