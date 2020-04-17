#!/usr/bin/python

from scipy.sparse import lil_matrix
from Util import split_row

# load the rating data as dok_matrix
def loadSparseR(usernum, itemnum, inFilePath):
	sR = lil_matrix((usernum, itemnum))
	with open(inFilePath, 'r') as infile:
		for line in infile.readlines():
			phs = split_row(line)
			if len(phs)==2:
				sR[int(phs[0]), int(phs[1])] = 1
			elif len(phs)==3:
				sR[int(phs[0]), int(phs[1])] = float(phs[2])
	return sR

def saveTriads(triads, outFilePath, isRatingInt=False):
	with open(outFilePath, 'w') as outfile:
		for ind in range(len(triads)):
			user, item, rating = triads[ind]
			if isRatingInt:
				outfile.write(str(int(user))+'\t'+str(int(item))+'\t%d\n' % rating)
			else:
				outfile.write(str(int(user))+'\t'+ str(int(item))+'\t%.1f\n' % rating)

if __name__ == '__main__':
	data_dir = '../../data/'

	# Movielens
	loadSparseR(2113, 10109, data_dir+'movielens/ml-2k/ratings_.txt')