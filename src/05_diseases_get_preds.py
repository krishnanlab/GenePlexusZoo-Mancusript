import utls
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score
import time
import argparse
import sys
import json


'''
This script will train a model for every disease using all labeled data (both
training and testing splits) and then save the prediction probability for every
gene.
'''


parser = argparse.ArgumentParser()
parser.add_argument('-g','--geneset_info',
                    default = 'MethodNum-0__DisNet_hs_20_200_0.5_0.5',
                    type = str,
                    help = 'information needed to make genesets')
parser.add_argument('-n','--network_info',
                    default = 'ce_hs__BioGRID_raw__eggnog_direct_SummedDegree-0.50__Pecanpy_2000_SparseOTF_0.1_0.001_120_8_10_weighted_directed.emb',
                    type = str,
                    help = 'information needed to get feature matrix')
parser.add_argument('-m','--model_info',
                    default = 'NoScaling__LR_l2_1.0',
                    type = str,
                    help = 'information on model and how to scale the data')
args = parser.parse_args()
geneset_info = args.geneset_info
network_info = args.network_info
model_info = args.model_info

print('The geneset_info is',geneset_info)
print('The network_info is',network_info)
print('The model_info is',model_info)

tic0 = time.time()
tic1 = time.time()
# This section will read in the network data
print('Reading in network data')
if '.adj' in network_info:
    FN_adj = '../data/features/AdjMat_Influence/'
    with open(FN_adj+network_info+'.json') as f:
      net_dict = json.load(f)
    print('The number of nodes in the network is',len(net_dict))
    
    all_genes_in_network = np.array(list(net_dict))
     
elif '.emb' in network_info:
    FN_net = '../data/features/embeddings/'+'%s'%network_info
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
print('The time in  minutes it took to read in network data is',int((time.time()-tic1)/60))

tic2 = time.time()
# This section will load the label_dict
fp_GSCs = '../data/GSCs/GSCs/'
with open(fp_GSCs+'MainLabels__%s_%s.json'%(geneset_info,network_info.split('__')[1].split('_')[0])) as f:
  label_dict = json.load(f)
print('The number of genesets in the label_dict is',len(label_dict))
print('The time in  minutes it took to read in label data is',int((time.time()-tic2)/60))

tic3 = time.time()
print('Doing ML part')
for idx, akey in enumerate(label_dict):
    #make the data and label arrays
    Xtrn = []
    ytrn = []
    for agene in label_dict[akey]['Train-Positive']:
        Xtrn.append(net_dict[agene])
        ytrn.append(1)
    for agene in label_dict[akey]['Train-Negative']:
        Xtrn.append(net_dict[agene])
        ytrn.append(0)
    for agene in label_dict[akey]['Test-Positive']:
        Xtrn.append(net_dict[agene])
        ytrn.append(1)
    for agene in label_dict[akey]['Test-Negative']:
        Xtrn.append(net_dict[agene])
        ytrn.append(0)
    Xtrn = np.array(Xtrn)
    ytrn = np.array(ytrn)
    num_trn_pos = len(label_dict[akey]['Train-Positive'])
    # scale data if desired
    if model_info.split('__')[0] == 'StdScl':
        std_scale = StandardScaler().fit(Xtrn)
        Xtrn   = std_scale.transform(Xtrn)
        Xtst   = std_scale.transform(Xtst)
    elif model_info.split('__')[0] == 'NoScaling':
        pass
    # set model parameters
    if model_info.split('__')[1].split('_')[0] == 'LR':
        mypen = model_info.split('__')[1].split('_')[1]
        myC = float(model_info.split('__')[1].split('_')[2])
        clf = LogisticRegression(max_iter=10000,solver='lbfgs',penalty=mypen,C=myC)
    # train the model
    clf.fit(Xtrn,ytrn)
    # get the predictions
    results_tmp = []
    all_genes = list(net_dict)
    chunks = [all_genes[x:x+1000] for x in range(0, len(all_genes), 1000)]
    for achunk in chunks:
        #make the array
        pred_data_tmp = []
        for agene in achunk:
            pred_data_tmp.append(net_dict[agene])
        pred_data_tmp = np.array(pred_data_tmp)
        probs_tmp = clf.predict_proba(pred_data_tmp)[:,1]
        probs_tmp = [item for item in probs_tmp]
        for idx, agene in enumerate(achunk):
            results_tmp.append([achunk[idx],probs_tmp[idx]])
    # save results
    col_names = ['Gene','Prob']
    df_results = pd.DataFrame(results_tmp,columns=col_names)
    save_dir = '../results/disease_preds/'
    df_results.to_csv(save_dir + 'MetricResults--%s--%s--%s--%s.tsv'%(args.geneset_info,args.network_info,args.model_info,akey),
                      sep='\t',header=True,index=False)
print('The time in  minutes it took to do ML is',int((time.time()-tic3)/60))
print('The time in  minutes it took to do the script is',int((time.time()-tic0)/60))
