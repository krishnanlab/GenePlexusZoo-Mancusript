import numpy as np
import pandas as pd
from itertools import combinations
import networkx as nx
import time
import glob
import scipy.linalg as sla
import json

tic = time.time()


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
    if '.adj' in network_info:
        FN_adj = '../data/features/AdjMat_Influence/'
        with open(FN_adj+network_info+'.json') as f:
          net_dict = json.load(f)
        print('The number of nodes in the network is',len(net_dict))
        all_genes_in_network = np.array(list(net_dict))
         
    elif '.emb' in network_info:
        FN_net = '../data/features/emneddings/'+'%s'%network_info
        net_dict = {}
        with open(FN_net,'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                line = line.strip().split()
                gene_tmp = line[0]
                emb_tmp = [float(item) for item in line[1:]]
                net_dict[gene_tmp] = emb_tmp
        print('The number of nodes in the network is',len(net_dict))
        print('The dimesions of the embeddings is',len(emb_tmp))
        all_genes_in_network = np.array(list(net_dict))
    return net_dict, all_genes_in_network

def get_genes_from_network(network,species):
    fp_net = '../data/node_degrees/'
    FN = fp_net + f'{network}-Summed_Degrees.tsv'
    dfgenes = pd.read_csv(FN,sep='\t',dtype=str)
    all_genes_in_network = dfgenes[dfgenes["Specie"]==species]["Gene"].to_numpy()
    print('The number of nodes in the network is',len(all_genes_in_network))
    return all_genes_in_network

def load_ortho_pairs(network,species):
    fp_ortho = "../data/edgelists/connections/"
    if species in ["ce","dm","dr"]:
        sp1 = species
        sp2 = "hs"
    else:
        sp1 = "hs"
        sp2 = species
    FN = fp_ortho + f"{sp1}_{sp2}__{network}_raw__direct__AllOnes.edgelist"
    dfortho = pd.read_csv(FN,sep="\t",header=None,names=[sp1,sp2,"weight"],dtype=str)
    return dfortho
       
