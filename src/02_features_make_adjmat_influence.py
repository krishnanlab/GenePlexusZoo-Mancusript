import pandas as pd
import numpy as np
import glob
import pickle
import random
import json
import argparse
import time
import scipy.linalg as sla
from itertools import combinations


'''
This script will save adjacency matrix and infulence matrix representations
of the networks as a json file.
'''

names_wormhole_dict = {'hs':'human','mm':'mouse','ce':'worm',
                       'dm':'fly','dr':'zebrafish','sc':'yeast'}
names_MGI_dict = {'hs':'Homo_sapiens','mm':'Mus_musculus','ce':'Caenorhabditis_elegans',
                  'dm':'Drosophila_melanogaster','dr':'Danio_rerio','sc':'Saccharomyces_cerevisiae'}
names_IMP_dict = {'hs':'human','mm':'mouse','ce':'celegans',
                  'dm':'fly','dr':'zebrafish','sc':'yeast'}
# there are two of these because of yeast in gene2pub and STRING are a bit different
names_pubmed_dict = {'hs':9606,'mm':10090,'ce':6239,
                    'dm':7227,'dr':7955,'sc':559292}
names_STRING_dict = {'hs':9606,'mm':10090,'ce':6239,
                    'dm':7227,'dr':7955,'sc':4932}


def read_network(network_info):
    tic0 = time.time()
    if '.adj' in network_info:
        net_combo = network_info.split('.ad')[0].split('__')
        data= []
        ID_map = {}
        current_node = 0
        # get the edgelist for each species
        for aspecies in net_combo[0].split('_'):
            if 'BioGRID' in net_combo[1]:
                FN_tmp = glob.glob('../data/edgelists/single_species/BioGRID/*%s*.edgelist'%names_MGI_dict[aspecies])[0]
                data, ID_map, current_node = read_edgelist(FN_tmp,data,ID_map,current_node,'undirected')
            elif 'IMP' in net_combo[1]:
                FN_tmp = glob.glob('../data/edgelists/single_species/IMP/%s.edgelist'%names_IMP_dict[aspecies])[0]
                data, ID_map, current_node = read_edgelist(FN_tmp,data,ID_map,current_node,'undirected')
        # add the connections
        if '_' in net_combo[0]: # if just one species this part is skipped
            fp_con = '../data/edgelists/connections/'
            specie_pairs = list(combinations(net_combo[0].split('_'),2))
            for a_pair in specie_pairs:
                FN_connect = fp_con+'%s_%s__%s__direct__%s.edgelist'%(a_pair[0],a_pair[1],net_combo[1],net_combo[3])
                if 'SummedDegree' in net_combo[3]:
                    data, ID_map, current_node = read_edgelist(FN_connect,data,ID_map,current_node,'directed')
                else:
                    data, ID_map, current_node = read_edgelist(FN_connect,data,ID_map,current_node,'undirected')
        print('The time it took to read in the edgelist is',time.time()-tic0)
        tic1 = time.time()
        net_dict, all_genes_in_network = make_net_dict(data,ID_map)
        print('The time it took to make the net dict is',time.time()-tic1)
        print('The number nodes and features in the adj mat dict is',len(net_dict),len(net_dict[list(net_dict)[0]]))
        
        
        if 'inf' in net_combo[1].split('_')[1]:
            tic2 = time.time()
            print('Making the influence matrix and resaving net dict')
            gene_order = list(net_dict)
            # make an adj mat
            adj_mat = np.zeros((len(gene_order),len(gene_order)),dtype=float)
            for idx, agene in enumerate(gene_order):
                adj_mat[idx,:] = net_dict[agene]
            # make influence matrix
            adj_mat = adj_mat / adj_mat.sum(axis=0)
            myidn = np.identity(adj_mat.shape[0])
            beta = float(net_combo[1].split('_')[1].split('-')[1])
            adj_mat = beta*sla.inv(myidn-(1-beta)*adj_mat)
            print('The time it took to make the influence matrix is',time.time()-tic2)
            if 'trans' in net_combo[1].split('_')[1]:
                tic3 = time.time()
                adj_mat = np.transpose(adj_mat)
                print('The time it took to do the transpose is',time.time()-tic3)
            # make into a new_dict again
            tic4 = time.time()
            for idx, agene in enumerate(gene_order):
                net_dict[agene] = list(adj_mat[idx,:])
            print('the time it took to remake the net dict is',time.time()-tic4)
            print('The number nodes and features in the inf mat dict is',len(net_dict),len(net_dict[list(net_dict)[0]]))
            
    return net_dict, all_genes_in_network
        
def read_edgelist(edgelist_file,data,ID_map,current_node,edgetype):
    with open(edgelist_file, 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) == 3:
                g1, g2, weight = line.strip().split('\t')
                weight = float(weight)
            elif len(line.strip().split('\t')) == 2:
                g1, g2 = line.strip().split('\t')
                weight = 1.0
            else:
                print('Not correct number of columns')
            if g1 not in ID_map:
                ID_map[g1] = current_node
                data.append({ID_map[g1]: 0})
                current_node += 1
            if g2 not in ID_map:
                ID_map[g2] = current_node
                data.append({ID_map[g2]: 0})
                current_node += 1
            if edgetype == 'undirected':
                data[ID_map[g1]][ID_map[g2]] = weight
                data[ID_map[g2]][ID_map[g1]] = weight
            elif edgetype == 'directed':
                data[ID_map[g1]][ID_map[g2]] = weight
    return data, ID_map, current_node
    
def make_net_dict(data,ID_map):
    all_genes_in_network = sorted(ID_map, key=ID_map.get)
    mysize = len(all_genes_in_network)
    net_dict = {}
    for idx, item in enumerate(data):
        myrowdata = np.zeros((mysize))
        col_inds = list(data[idx].keys())
        col_weights = list(data[idx].values())
        myrowdata[col_inds] = col_weights
        myrowdata = list(myrowdata)
        net_dict[all_genes_in_network[idx]] = myrowdata
    return net_dict, all_genes_in_network
    
def save_net_dict(net_dict,network_info):
    tic5 = time.time()
    fp_save = '../data/features/AdjMat_Influence/'
    net_split = network_info.split("__")
    with open(fp_save+'%s__%s__eggnog_direct_%s.json'%(net_split[0],net_split[1],
                                     net_split[3]),'w') as handle:
        json.dump(net_dict, handle, sort_keys=True, indent=4)
    print('the time it took to save json is',time.time()-tic5)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--network_info',
                        default = 'ce_hs__BioGRID_raw__direct__AllOnes.adj',
                        type = str,
                        help = 'information needed to get feature matrix')
    args = parser.parse_args()
    network_info = args.network_info
    
    print()
    print('The network_info is',network_info)
    net_dict, all_genes_in_network = read_network(network_info)
    save_net_dict(net_dict,network_info)

