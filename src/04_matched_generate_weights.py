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
import glob


'''
This script will use all the labeled data (both traiing and testing splits) to
train a model and then the model weights are saved.

Note: This is only done for adjacnecy matrix representations using the AllOnes
weighting scheme and the peconypy representation utilized throughout the work.
'''

parser = argparse.ArgumentParser()
parser.add_argument('-g','--gsc',
                    default = 'GO_ce',
                    type = str,
                    help = 'the gsc')
parser.add_argument('-s','--species',
                    default = 'ce_hs',
                    type = str,
                    help = 'the gsc')
parser.add_argument('-n','--network',
                    default = 'BioGRID',
                    type = str,
                    help = 'BioGRID or IMP')
parser.add_argument('-f','--features',
                    default = 'emb',
                    type = str,
                    help = 'emb or adj or allemb')
args = parser.parse_args()
gsc = args.gsc
species = args.species
network = args.network
features = args.features

print('The gsc is',gsc)
print('The species are',species)
print('The network is',network)
print('The features are',features)

tic0 = time.time()
tic1 = time.time()

if network == "BioGRID":
    OTFtype = "Sparse"
elif network == "IMP":
    OTFtype = "Dense"

# This section will read in the network data
print('Reading in network data')
if features == 'adj':
    FN_adj = '../data/features/AdjMat_Influence/'
    FN_end = f"{species}__{network}_raw__eggnog_direct_AllOnes.adj"
    with open(FN_adj+FN_end+'.json') as f:
      net_dict = json.load(f)
    print('The number of nodes in the network is',len(net_dict))

    all_genes_in_network = np.array(list(net_dict))
elif (features == 'emb') or (features == "allemb"):
    if features == "emb":
        sp_feats = species
    elif features == "allemb":
        sp_feats = "ce_dm_dr_hs_mm_sc"
    FN_net = '../data/features/embeddings/'
    FN_end = f"{sp_feats}__{network}_raw__eggnog_direct_SummedDegree-0.50__Pecanpy_2000_{OTFtype}OTF_0.1_0.001_120_8_10_weighted_directed.emb"
    net_dict = {}
    with open(FN_net+FN_end,'r') as f:
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
fp_GSCs = '../data/GSCs/GSCsBothTerms/'
fp_end = f"MainLabels__MethodNum-0__{gsc}__{species}__15_200_0.5_0.5__{network}.json"
with open(fp_GSCs+fp_end) as f:
  label_dict = json.load(f)
print('The number of genesets in the label_dict is',len(label_dict))
print('The time in  minutes it took to read in label data is',int((time.time()-tic2)/60))

tic3 = time.time()
weights_dict = {}
print('Doing ML part')
for idx, akey in enumerate(label_dict):
    weights_dict[akey] = {}
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

    clf = LogisticRegression(max_iter=10000,solver='lbfgs',penalty='l2',C=1.0)
    clf.fit(Xtrn,ytrn)
    # get the weights
    mycoef = clf.coef_.astype('float32')
    myint = clf.intercept_.astype('float32')
    weights_dict[akey]['weights'] = mycoef.tolist()[0]
    weights_dict[akey]['intercept'] = float(myint)

save_dir = '../results/bothterms_weights/'
with open(save_dir+f'{gsc}__{species}__{network}__{features}.json','w') as f:
    json.dump(weights_dict,f,ensure_ascii=False,indent=4)

