
import warnings
warnings.filterwarnings("ignore")


import sys
sys.path.append("./models/")
from ensemble_ import Ensemble
sys.path.append("../../utils/")
from IOUtil import loadSparseR
from Util import matBinarize
sys.path.append("../../samplers/")
from sampler_ranking import Sampler

import multiprocessing
from scipy.sparse import lil_matrix
import numpy as np

data_dir = '../../../data/'
folds = 5
binarize_threshold = 3

reg=.1
kensemble = 5

topN = 10
split_method='cv'
eval_metrics = ['pre','recall', 'map', 'mrr', 'ndcg']
n_factors=100
batch_size=100
negSample=5

def worker(fold, n_users, n_items, dataset_dir):
    traFilePath = dataset_dir + 'ratings__' + str(fold + 1) + '_tra.txt'
    trasR = lil_matrix(matBinarize(loadSparseR(n_users, n_items, traFilePath), binarize_threshold))

    print(dataset_dir.split('/')[-2] + '@%d:' % (fold + 1), trasR.shape, trasR.nnz,
          '%.2f' % (trasR.nnz / float(trasR.shape[0])))

    tstFilePath = dataset_dir + 'ratings__' + str(fold + 1) + '_tst.txt'
    tstsR = lil_matrix(matBinarize(loadSparseR(n_users, n_items, tstFilePath), binarize_threshold))

    sampler = Sampler(trasR=trasR, n_neg=negSample, batch_size=batch_size)

    en = Ensemble(n_users, n_items, kensemble, topN, split_method, eval_metrics, reg, n_factors, batch_size)

    scores = en.train(fold+1, trasR, tstsR, sampler)

    print(dataset_dir.split('/')[-2] + '@%d:' % (fold + 1),
          ','.join(['%s' % eval_metric for eval_metric in eval_metrics]) + '@%d=' % (topN) +
          ','.join(['%.6f' % (score) for score in scores]))

    en.close()
    return scores


if __name__ =='__main__':
    print('reg=',reg, 'kensemble=', kensemble)

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

    # yelp
    # dataset_dir = data_dir+'yelp/'
    # n_users, n_items = 16239, 14284

    # douban book
    # dataset_dir = data_dir+'douban/book/'
    # n_users, n_items = 13024, 22347

    # Amazon
    # dataset_dir = data_dir+'Amazon/'
    # n_users, n_items = 6170, 2753

    # douban Movie_
    # dataset_dir = data_dir+'douban/Movie_/'
    # n_users, n_items = 13367, 12677

    # yelp
    # dataset_dir = data_dir+'yelp/'
    # n_users, n_items = 16239, 14284

    # yelp
    # dataset_dir = data_dir+'yelp/yelp_data(u14085b14037)/'
    # n_users, n_items = 14085, 14037

    # yelp-200k
    # dataset_dir = data_dir+'yelp/yelp-200k/'
    # n_users, n_items = 36105, 22496

    # douban book
    # dataset_dir = data_dir+'douban/db_book_u12850b22040/'
    # n_users, n_items = 11777, 20697

    # dianping
    # dataset_dir = data_dir+'dianping/dp_dataset_u10549s17707/'
    # n_users, n_items = 10549, 17707

    # process pool
    pool = multiprocessing.Pool(processes=1)
    results = []
    for fold in range(1):
        results.append(pool.apply_async(func=worker, args=(fold, n_users, n_items, dataset_dir)))
    pool.close()
    pool.join()
    pool.terminate()

    # scores = np.array([result.get() for result in results])
    # aves = scores.sum(0)/len(scores)
    # stds = np.sqrt(np.power(np.array(scores) - aves,2).sum(0)/len(scores))
    # print('ave@'+str(topN)+'=['+','.join(['%.4f' %ave for ave in aves])+']', 'std@'+str(topN)+'=['+','.join(['%.4f' %std for std in stds])+']')
