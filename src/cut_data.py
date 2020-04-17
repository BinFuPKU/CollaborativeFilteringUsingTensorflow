
import numpy as np

import sys
sys.path.append("./utils/")
from IOUtil import loadSparseR, saveTriads


# k-fold cross validation.
def split_tra_tst(usernum, itemnum, inFilePath, cv_fold):
	sR=loadSparseR(usernum, itemnum, inFilePath)
	useritemrating_tuples = np.array([(user,item,sR[user,item]) for user,item in np.asarray(sR.nonzero()).T])
	np.random.shuffle(useritemrating_tuples)

	fold_ins_num = int(len(useritemrating_tuples) / cv_fold)
	outdir, filename = '/'.join(inFilePath.split('/')[:-1])+'/', inFilePath.split('/')[-1]
	for cv_fold_ind in range(cv_fold):
		tst_outFilePath = outdir + filename.replace('.', '_'+str(cv_fold_ind+1)+'_tst.')
		tst = useritemrating_tuples[cv_fold_ind*fold_ins_num: (cv_fold_ind+1)*fold_ins_num,:]
		saveTriads(tst, tst_outFilePath, isRatingInt=False)

		tra_outFilePath = outdir + filename.replace('.', '_'+str(cv_fold_ind+1)+'_tra.')
		tra = np.concatenate([useritemrating_tuples[: cv_fold_ind*fold_ins_num,:],
								  useritemrating_tuples[(cv_fold_ind+1)*fold_ins_num:,:]])
		saveTriads(tra, tra_outFilePath, isRatingInt=False)

if __name__ == '__main__':
	data_dir = '../data/'
	folds = 5

	# Movielens
	# ml-100k -
	# dataset_dir = data_dir+'movielens/ml-100k/'
	# n_users, n_items = 943, 1682
	# ml-1m -
	# n_users, n_items = 6040, 3706
	# split_tra_tst(n_users, n_items, data_dir+'movielens/ml-1m/ratings_.txt', folds)
	# ml-2k -
	# n_users, n_items = 2113, 10109
	# split_tra_tst(n_users, n_items, data_dir+'movielens/ml-2k/ratings_.txt', folds)
	# ml-20m
	# n_users, n_items = 138493, 26744
	# split_tra_tst(n_users, n_items, data_dir+'movielens/ml-20m/ratings_.txt', folds)

	# lastfm2k
	# n_users, n_items = 1892, 17632
	# split_tra_tst(n_users, n_items, data_dir+'lastfm/lastfm2k/ratings_.txt', folds)

	# netflix5k5k
	# n_users, n_items = 4744, 3550
	# split_tra_tst(n_users, n_items, data_dir+'netflix5k5k/ratings_.txt', folds)

	# CiaoDVD
	# n_users, n_items = 21019, 71633
	# split_tra_tst(n_users, n_items, data_dir+'CiaoDVD/rratings_.txt', folds)

	# douban book -
	# n_users, n_items = 13024, 22347
	# split_tra_tst(n_users, n_items, data_dir+'douban/book/ratings_.txt', folds)

	# douban movie -
	# n_users, n_items = 6170, 2753
	# split_tra_tst(n_users, n_items, data_dir+'Amazon/ratings_.txt', folds)

	# douban Movie_
	n_users, n_items = 13367, 12677
	split_tra_tst(n_users, n_items, data_dir+'douban/Movie_/ratings_.txt', folds)

	# yelp  -
	# n_users, n_items = 14085, 14037
	# split_tra_tst(n_users, n_items, data_dir+'yelp/yelp_data(u14085b14037)/ratings_.txt', folds)

	# yelp -
	# n_users, n_items = 16239, 14284
	# split_tra_tst(n_users, n_items, data_dir+'yelp/ratings_.txt', folds)

	# yelp
	# n_users, n_items = 36105, 22496
	# split_tra_tst(n_users, n_items, data_dir+'yelp/yelp-200k/ratings_.txt', folds)

	# douban book
	# n_users, n_items = 11777, 20697
	# split_tra_tst(n_users, n_items, data_dir+'douban/db_book_u12850b22040/ratings_.txt', folds)

	# dianping -
	# n_users, n_items = 10549, 17707
	# split_tra_tst(n_users, n_items, data_dir+'dianping/dp_dataset_u10549s17707/ratings_.txt', folds)

	# filmtrust -
	# n_users, n_items = 1508, 2071
	# split_tra_tst(n_users, n_items, data_dir+'filmtrust/ratings_.txt', folds)

	# wikilens
	# n_users, n_items = 326, 5111
	# split_tra_tst(n_users, n_items, data_dir+'wikilens/ratings_.txt', folds)