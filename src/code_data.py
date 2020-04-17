
# code the data int index from 0,1,2,...
import sys
sys.path.append("./utils/")
from Util import split_row


def code_data(inFilePath):
	filename = inFilePath.split('/')[-1]
	outFilePath = '/'.join(inFilePath.split('/')[:-1])+'/'+filename.replace('.', '_.')
	user_dict, item_dict, instance_num = {}, {}, 0

	print('code data from',inFilePath,'to',outFilePath)

	scores = set()
	with open(inFilePath, 'r') as inFile, open(outFilePath, 'w') as outFile:
		for line in inFile.readlines():
			phs = split_row(line)
			if int(phs[0]) not in user_dict:
				user_dict[int(phs[0])] = len(user_dict)
			if int(phs[1]) not in item_dict:
				item_dict[int(phs[1])] = len(item_dict)
			line = str(user_dict[int(phs[0])])+'\t'+ str(item_dict[int(phs[1])])+'\t'
			if len(phs)>2:
				line += phs[2]
				scores.add(phs[2])
			else:
				line += '1'
			outFile.write(line+'\n')
			instance_num += 1
	print('stat: user_num=',len(user_dict),'item_num=',len(item_dict),'instance_num=',instance_num,
		  'density=%.4f' % (instance_num/float(len(user_dict) * len(item_dict))))
	print(scores)

if __name__ == '__main__':
	data_dir = '../data/'

	# Movielens
	# code_data(data_dir+'movielens/ml-20m/ratings.txt') # ml-2k

	# lastfm
	# code_data(data_dir+'lastfm/lastfm2k/ratings.txt')

	# netflix5k5k
	# code_data(data_dir+'netflix5k5k/ratings.txt')

	# yelp
	# code_data(data_dir+'yelp/ratings.txt')

	# Epinions
	# code_data(data_dir+'Epinions/Epinions665K/ratings.txt')

	# eachmovie
	# code_data(data_dir+'eachmovie/ratings.txt')

	# delicious2k
	# code_data(data_dir+'delicious/delicious2k/ratings.txt')

	# CiaoDVD
	# code_data(data_dir+'CiaoDVD/rratings.txt')

	# BookCrossing
	# code_data(data_dir+'BookCrossing/ratings.txt')

	# douban book
	# code_data(data_dir+'douban/book/ratings.txt')

	# Amazon
	# code_data(data_dir+'Amazon/ratings.txt')

	# douban movie
	# code_data(data_dir+'douban/movie/ratings.txt')

	# douban movie
	# code_data(data_dir+'douban/Movie_/ratings.txt')

	# yelp
	# code_data(data_dir+'yelp/yelp_data(u14085b14037)/ratings.txt')

	# yelp
	# code_data(data_dir+'yelp/yelp-200k/ratings.txt')

	# douban book
	# code_data(data_dir+'douban/db_book_u12850b22040/ratings.txt')

	# dianping
	# code_data(data_dir+'dianping/dp_dataset_u10549s17707/ratings.txt')

	# filmtrust
	# code_data(data_dir+'filmtrust/ratings.txt')

	# wikilens
	code_data(data_dir+'wikilens/ratings.txt')