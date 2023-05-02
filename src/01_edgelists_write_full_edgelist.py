import pandas as pd
import numpy as np
from itertools import combinations, product
import time
import argparse
import glob

'''
This script is used to write the edgelist to a file so it can be read by 
the node2vec script.
'''

if __name__ == '__main__':
    
    tic0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--info',
                        default = 'ce_dr__BioGRID_raw__direct__AllOnes',
                        type = str,
                        help = 'what data should be used to build the edgelist')
    args = parser.parse_args()
    info = args.info
    
    fp_save ='../data/edgelists/full/'
    # check to see if edgelist exisits and if so skip
    done_files = glob.glob(fp_save+'*.edgelist')
    if fp_save + info + '.edgelist' in done_files:
        print('File for',info,'already exsists')
        pass
    else:    
        print('The file being made is',info)

        # inizalize of dataframe to put all the data
        df_final = pd.DataFrame()
        # split up the argparse input
        net_combo = info.strip().split('__')
        # set the base directory where the files can be found
        fp_main_net = '../data/edgelists/'
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


        print('Concatenating together the edge from single specie files')
        for aspecies in net_combo[0].split('_'):
            if 'BioGRID' in net_combo[1]:
                FN_single = glob.glob('../data/edgelists/single_species/BioGRID/*%s*.edgelist'%names_MGI_dict[aspecies])[0]
                df_tmp = pd.read_csv(FN_single,sep='\t', header=None, names = ['Gene1','Gene2'],dtype=str)
                df_tmp['Weight'] = df_tmp.shape[0] * ['1.0']
            elif 'IMP' in net_combo[1]:
                FN_single = glob.glob('../data/edgelists/single_species/IMP/%s.edgelist'%names_IMP_dict[aspecies])[0]
                df_tmp = pd.read_csv(FN_single,sep='\t', header=None, names = ['Gene1','Gene2','Weight'],dtype=str)
            else:
                print('Unknown network info')
            df_final = pd.concat([df_final,df_tmp],ignore_index=True)
            # add directed edges to single species networks
            if 'SummedDegree' in net_combo[3]:
                df_tmp.columns = ['Gene2','Gene1','Weight']
                df_tmp = df_tmp[['Gene1','Gene2','Weight']]
                df_final = pd.concat([df_final,df_tmp],ignore_index=True)


        print('Starting to add the network connections')
        # add on the connections
        if '_' in net_combo[0]: # if just one species this part is skipped
            fp_con = '../data/edgelists/connections/'
            FN_connect = fp_con+'%s__%s__%s__%s.edgelist'%(net_combo[0],net_combo[1],
                                                           net_combo[2],net_combo[3])
            df_tmp = pd.read_csv(FN_connect,sep='\t', header=None, names = ['Gene1','Gene2','Weight'])
            df_final = pd.concat([df_final,df_tmp],ignore_index=True)


        print('Starting to write the file')
        df_final.to_csv(fp_save + info + '.edgelist', sep='\t', header=False, index=False,float_format='%.3f')
        print('The shape of the final file is', df_final.shape)
    print('The time it took to run this script is', time.time()-tic0)
