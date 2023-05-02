import argparse
import pandas as pd
import numpy as np
import pickle
import time
import sys
import datetime
import glob
import itertools

'''
This script will load networks from two species and the egnnog ortholog database
and then connect gene across species for a specified weighting scheme
'''


def load_networks(species1, species2, network_type):    
    if network_type == 'BioGRID_raw':
        FN1 = glob.glob('../data/edgelists/single_species/BioGRID/*%s*.edgelist'%names_MGI_dict[species1])[0]
        df1 = pd.read_csv(FN1,sep='\t', header=None, names = ['Gene1','Gene2'],dtype=str)
        FN2 = glob.glob('../data/edgelists/single_species/BioGRID/*%s*.edgelist'%names_MGI_dict[species2])[0]
        df2 = pd.read_csv(FN2,sep='\t', header=None, names = ['Gene1','Gene2'],dtype=str)
    elif network_type == 'IMP_raw':
        FN1 = glob.glob('../data/edgelists/single_species/IMP/%s.edgelist'%names_IMP_dict[species1])[0]
        df1 = pd.read_csv(FN1,sep='\t', header=None, names = ['Gene1','Gene2','Weight'],dtype=str)
        FN2 = glob.glob('../data/edgelists/single_species/IMP/%s.edgelist'%names_IMP_dict[species2])[0]
        df2 = pd.read_csv(FN2,sep='\t', header=None, names = ['Gene1','Gene2','Weight'],dtype=str)
    else:
        print('Unknown network info')
    return df1, df2
    
def get_unique_genes(df1, df2):
    df1_genes = np.union1d(df1['Gene1'].unique(),df1['Gene2'].unique())
    df2_genes = np.union1d(df2['Gene1'].unique(),df2['Gene2'].unique())
    return df1_genes, df2_genes
    
def load_mapping_file():
    fp_ogs = '../data/eggnog/'
    with open(fp_ogs + 'EGGNOG-6species-Entrez_Mappings.pickle', 'rb') as handle:
        mapping_file = pickle.load(handle)        
    return mapping_file
    
def get_degree_dicts(network,s1,s2):
    fp_degrees = '../data/node_degrees/'
    s1_deg_dict = {}
    s2_deg_dict = {}
    with open(fp_degrees + '%s-Summed_Degrees.tsv'%network.split('_')[0], 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip().split('\t')
            if line[2] == s1:
                s1_deg_dict[line[0]] = float(line[1])
            if line[2] == s2:
                s2_deg_dict[line[0]] = float(line[1])    
    return s1_deg_dict, s2_deg_dict
    
def get_count_dict(results):
    gene_cnt_dict = {}
    for item in results:
        if item[0] not in gene_cnt_dict:
            gene_cnt_dict[item[0]] = 1
        else:
            gene_cnt_dict[item[0]] = gene_cnt_dict[item[0]] + 1
        if item[1] not in gene_cnt_dict:
            gene_cnt_dict[item[1]] = 1
        else:
            gene_cnt_dict[item[1]] = gene_cnt_dict[item[1]] + 1
    return gene_cnt_dict
    
if __name__ == '__main__':
    
    tic = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_type','-nt',
                        default = 'BioGRID_raw', 
                        type = str,
                        help = 'The type of network')
    parser.add_argument('--species1','-s1',
                        default = 'ce', 
                        type = str,
                        help = 'The first species')
    parser.add_argument('--species2','-s2',
                        default = 'hs', 
                        type = str,
                        help = 'The second species')
    parser.add_argument('--weights','-w',
                        default = 'AllOnes', 
                        type = str,
                        help = 'The way to add the connections')
    args = parser.parse_args()    
    network_type = args.network_type
    species1 = args.species1
    species2 = args.species2
    connection = "direct"
    weights = args.weights
    datatype = "eggnog"
    # connection and datatype were tried in early iterations of the method
    # but are no longer considered as hyperparameters
    
    fp_end = '%s_%s__%s__%s__%s'%(species1,species2,network_type,connection,weights)
    
    # make some name conversion dictionaries
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
    
    print('Running for %s %s %s %s %s'%(species1,species2,network_type,connection,weights))
    print()
    print('Loading the networks')
    df1,df2 = load_networks(species1,species2,network_type)
    print('The number of edges in %s network is'%species1,df1.shape[0])
    print('The number of edges in %s network is'%species2,df2.shape[0])
    print()
    
    print('Finding the unique genes in the networks')
    df1_genes, df2_genes = get_unique_genes(df1, df2)
    print('The number of unique genes in %s network is'%species1,df1_genes.shape[0])
    print('The number of unique genes in %s network is'%species2,df2_genes.shape[0])
    print()
    
    print('Loading mapping_file file')
    mapping_file = load_mapping_file()
    
    print('Making cross-species connections')
    group_cnt = 0
    gene1_cnt = 0
    gene2_cnt = 0
    results = []
    genes1_unq = []
    genes2_unq = []
    for idx, akey in enumerate(mapping_file):
        mapping_genes = np.array(mapping_file[akey])
        genes1 = np.intersect1d(df1_genes,mapping_genes)
        genes2 = np.intersect1d(df2_genes,mapping_genes)
        if (len(genes1) > 0) and (len(genes2) > 0):
            # add a running list of the genes that had at least one connection
            genes1_unq = list(set(list(genes1)+genes1_unq))
            genes2_unq = list(set(list(genes2)+genes2_unq))
            # count number of good OGs and genes
            group_cnt = group_cnt + 1
            gene1_cnt = gene1_cnt + len(genes1)
            gene2_cnt = gene2_cnt + len(genes2)
            # start adding the connections
            if  connection == 'groups':
                group_name = akey
                for agene1 in genes1:
                    results.append([group_name,agene1])
                for agene2 in genes2:
                    results.append([group_name,agene2])
            elif connection == 'direct':
                for agene1 in genes1:
                    for agene2 in genes2:
                        results.append([agene1,agene2])
        else:
            continue
            
    print('The number of groups that had genes in both species is',group_cnt)
    print('The number of genes in %s that had one cross-species connection is'%species1,len(genes1_unq))
    print('The number of genes in %s that had one cross-species connection is'%species2,len(genes2_unq))
    if connection == 'groups':
        print('The number of %s connection made to a group is'%species1,gene1_cnt)
        print('The number of %s connection made to a group is'%species2,gene2_cnt)
    elif connection == 'direct':
        print('The number of %s to %s connections added is'%(species1,species2),len(results))
    print()
    
    print('Saving the connecitons file')
    if weights == 'AllOnes':
        results = [item + ['1.0'] for item in results]
    elif weights == 'AllTwos':
        results = [item + ['2.0'] for item in results]
    elif weights == 'AllFives':
        results = [item + ['5.0'] for item in results]
    elif 'SummedDegree' in weights:
        percent = float(weights.strip().split('-')[1])
        s1_deg_dict, s2_deg_dict = get_degree_dicts(network_type,species1,species2)
        # remove duplicate edges
        results = sorted(results)
        results = [results[i] for i in range(len(results)) if i == 0 or results[i] != results[i-1]]
        gene_cnt_dict = get_count_dict(results)
        results_new = []
        for aline in results:
            # this sets how much total probabilty it will walk to the other species
            g1_weight = s1_deg_dict[aline[0]] * percent
            g2_weight = s2_deg_dict[aline[1]] * percent
            # this dived the total probably across all connection to other species
            g1_weight = g1_weight / gene_cnt_dict[aline[0]]
            g2_weight = g2_weight / gene_cnt_dict[aline[1]]
            results_new.append([aline[0],aline[1],g1_weight])
            results_new.append([aline[1],aline[0],g2_weight])
        results = results_new

    else:
        print('A unknown type was called so no weight column added')
        dfsdfds
    df_final = pd.DataFrame(results)
    df_final = df_final.drop_duplicates()
    print('The shape of the final df is',df_final.shape)
    # Need to be careful with the number of float digits if not setting weights to 1.0
    df_final.to_csv('../data/edgelists/connections/%s.edgelist'%fp_end,sep='\t',index=False,
                                                            header=False,float_format='%.3f')
    
        
    print('The time is took to run this script is',time.time()-tic)



