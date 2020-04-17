
import numpy as np

# split the row content
def split_row(row_content):
	if ',' in row_content:
		return row_content.strip().split(',')
	if ';' in row_content:
		return row_content.strip().split(';')
	else:
		return row_content.strip().split()


# binary (sparse matrix)
def matBinarize(sR, r_threshold):
	return (sR>r_threshold).astype(np.float32)
