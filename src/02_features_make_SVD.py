import numpy as np
import pandas as pd
import argparse
import time
import utls
from sklearn.decomposition import TruncatedSVD
import scipy.linalg as sla

'''
This script will load the adjacency matrix file and perform SVD on it.
'''

parser = argparse.ArgumentParser()
parser.add_argument('-n','--network_info',
                    default = 'ce_hs__BioGRID_raw__eggnog_direct_AllOnes',
                    type = str,
                    help = 'information needed to get feature matrix')
parser.add_argument('-s','--scaling',
                    default = 'StdScl',
                    type = str,
                    help = 'How to scale the matrix : (None, Scl or StdScl)')
parser.add_argument('-d','--num_dims',
                    default = 500,
                    type = int,
                    help = 'Number of dimensions')
args = parser.parse_args()
network_info = args.network_info
network_info_adj = network_info + '.adj'
scaling = args.scaling
num_dims = args.num_dims


tic0 = time.time()

# This section will read in the network data
print('Reading in network data')
time_tmp = time.time()
net_dict, all_genes_in_network = utls.read_network(network_info_adj)
print('The number of seconds it took to read edgelist in is', time.time()-time_tmp)

# turn the dictionary into a matrix
time_tmp = time.time()
arr_size = len(all_genes_in_network)
adj_mat = np.zeros((arr_size,arr_size),dtype=float)
for idx, agene in enumerate(all_genes_in_network):
    adj_mat[idx,:] = net_dict[agene]
print('The number of seconds it took to change dict into array is', time.time()-time_tmp)

time_tmp = time.time()
# scale the data into the model
if scaling == 'None':
    pass
elif scaling == 'Scl':
    adj_mat = adj_mat - np.mean(adj_mat,axis=0)
elif scaling == 'StdScl':
    adj_mat = (adj_mat - np.mean(adj_mat,axis=0)) / np.std(adj_mat,axis=0)
else:
    print('A wrong sclaing was put in')
    hjhkjhkgg
print('The number of seconds it took to scale array is', time.time()-time_tmp)

# set SVD and hard code some arguments

time_tmp = time.time()
svd = TruncatedSVD(n_components=num_dims, n_iter=5, random_state=42, algorithm='randomized')
adj_mat_reduced = svd.fit_transform(adj_mat)
print('The number of seconds it took to change dict into array is', time.time()-time_tmp)
print('The shape of the embedded array is',adj_mat_reduced.shape)
print('The variance explained by this embedding is',np.sum(svd.explained_variance_ratio_))


print('Saving the embedding')
save_dir = '../data/features/embeddings/'
with open(save_dir + '%s__SVD_%s_%s.emb'%(network_info,num_dims,scaling), 'w') as thefile:
    thefile.write("%s %s\n" %(adj_mat_reduced.shape[0],adj_mat_reduced.shape[1]))
    for idx, agene in enumerate(all_genes_in_network):
        next_line = '%s '%agene + ' '.join(['%.6f'%item for item in adj_mat_reduced[idx,:]])
        thefile.write("%s\n" %next_line)

print('The number of seconds it took to run the script is', time.time()-tic0)
