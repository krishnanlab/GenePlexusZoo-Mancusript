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
This script will generate a file with connections between more than one species.
Note: If a SummedDegree weighting scheme has to be performed, then the AllOnes files from
01_edgelists_pairwise_connections.py must be generated first.
'''


def load_networks(aspecie, network_type):    
    if network_type == 'BioGRID_raw':
        FN = glob.glob('../data/edgelists/single_species/BioGRID/*%s*.edgelist'%names_MGI_dict[aspecie])[0]
        df = pd.read_csv(FN,sep='\t', header=None, names = ['Gene1','Gene2'],dtype=str)
    elif network_type == 'IMP_raw':
        FN = glob.glob('../data/edgelists/single_species/IMP/%s.edgelist'%names_IMP_dict[aspecie])[0]
        df = pd.read_csv(FN,sep='\t', header=None, names = ['Gene1','Gene2','Weight'],dtype=str)
    else:
        print('Unknown network info')
    return df
    
def get_unique_genes(df):
    df_genes = np.union1d(df['Gene1'].unique(),df['Gene2'].unique())
    return df_genes
    
def load_mapping_file():
    fp_ogs = '../data/eggnog/'
    with open(fp_ogs + 'EGGNOG-6species-Entrez_Mappings.pickle', 'rb') as handle:
        mapping_file = pickle.load(handle)        
    return mapping_file
    
def concat_cons(df_cons,fp_end_con):
    df_tmp = pd.read_csv('../data/edgelists/connections/%s.edgelist'%fp_end_con,
                         sep='\t',header=None,names=['Gene1','Gene2','Weight'])
    df_cons = pd.concat([df_cons,df_tmp])
    return df_cons
    
    
def get_degree_dicts(network,species_list):
    fp_degrees = '../data/node_degrees/'
    deg_dict = {}
    with open(fp_degrees + '%s-Summed_Degrees.tsv'%network.split('_')[0], 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip().split('\t')
            if line[2] in species_list:
                deg_dict[line[0]] = float(line[1])
    return deg_dict
    
def get_count_dict(df_cons,all_genes):
    gene_cnt_dict = {}
    for agene in all_genes:
        num_cons = df_cons[(df_cons['Gene1']==agene) | (df_cons['Gene2']==agene)].shape[0]
        gene_cnt_dict['%i'%agene] = num_cons
    return gene_cnt_dict
    
if __name__ == '__main__':
    
    tic = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_type','-nt',
                        default = 'BioGRID_raw', 
                        type = str,
                        help = 'The type of network')
    parser.add_argument('--species','-s',
                        default = 'ce_dr_mm', 
                        type = str,
                        help = 'species need to be alpha order')
    parser.add_argument('--weights','-w',
                        default = 'SummedDegree-0.75', 
                        type = str,
                        help = 'The way to add the connections')
    args = parser.parse_args()    
    network_type = args.network_type
    species = args.species
    weights = args.weights
    datatype = "eggnog"
    connection = "direct"
    # Datatype and connection were used in earlier versions of the code
    # but are no longer considered
    
    fp_end = '%s__%s__%s__%s'%(species,network_type,connection,weights)
    
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
    
    print('Running for %s %s %s %s'%(species,network_type,connection,weights))
    print()
    print('Convert the species into a list')
    species_list = species.strip().split('_')
    print()
    
    print('Loading the networks')
    data_dict = {}
    for aspecie in species_list:
        data_dict[aspecie] = {}
        df = load_networks(aspecie,network_type)
        df_genes = get_unique_genes(df)
        data_dict[aspecie]["edgelist"] = df
        data_dict[aspecie]["unique_genes"] = df_genes
        print('The number of edges in %s network is'%aspecie,df.shape[0])
        print('The number of unique genes in %s network is'%aspecie,df_genes.shape[0])
    print()
    
    print('Loading mapping_file file')
    mapping_file = load_mapping_file()
    print('The number of groups in the mapping file is',len(list(mapping_file)))
    print()

    print('Concatenating pairwise cross-species connections')
    specie_combos = list(itertools.combinations(species_list,2))
    df_cons = pd.DataFrame()
    for apair in specie_combos:
        if weights in ['AllOnes','AllTwos','AllFives']:
            fp_end_con = '%s_%s__%s__%s__%s'%(apair[0],apair[1],network_type,connection,weights)
        elif 'SummedDegree' in weights:
            fp_end_con = '%s_%s__%s__%s__AllOnes'%(apair[0],apair[1],network_type,connection)
        df_cons = concat_cons(df_cons,fp_end_con)
    df_cons = df_cons.reset_index()
    all_genes = np.union1d(df_cons['Gene1'].unique(),df_cons['Gene2'].unique())
    print('The total number of edges present after concatenation is',df_cons.shape)
    print('The total number of unique genes after concatenation is',len(all_genes))
    print()

    print('Generating weights and saving the connecitons file')
    if weights in ['AllOnes','AllTwos','AllFives']:
        pass
    elif 'SummedDegree' in weights:
        percent = float(weights.strip().split('-')[1])
        deg_dict = get_degree_dicts(network_type,species_list)
        gene_cnt_dict = get_count_dict(df_cons,all_genes)
        results_new = []
        for index, row in df_cons.iterrows():
            gene1 = '%i'%row['Gene1']
            gene2 = '%i'%row['Gene2']
            # this sets how much total probability it will walk to the other specie
            g1_weight = deg_dict[gene1] * percent
            g2_weight = deg_dict[gene2] * percent
            # this divides the total probability across all connection to other specie
            g1_weight = g1_weight / gene_cnt_dict[gene1]
            g2_weight = g2_weight / gene_cnt_dict[gene2]
            # add edges in both directions
            results_new.append([gene1,gene2,g1_weight])
            results_new.append([gene2,gene1,g2_weight])
        df_cons = pd.DataFrame(results_new)
    else:
        print('An unknown type was called so no weight column added')
        dfsdfds
    df_cons = df_cons.drop_duplicates()
    print('The shape of the final df is',df_cons.shape)
    # # Need to be careful with the number of float digits if not setting weights to 1.0
    df_cons.to_csv('../data/edgelists/connections/%s.edgelist'%fp_end,sep='\t',index=False,
                                                  header=False,float_format='%.3f')

        
    print('The time it took to run this script is',time.time()-tic)



