import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score
import argparse
import time


'''
This script will load the the model coefficients generated in matched_generate_weights.py
and evaluate the model using the other species annotations.

Note this script breaks it down to only due a few terms at a time, so in order
to get all terms need to be run different sets of terms separately then combine
the results into one file at the end.
'''

parser = argparse.ArgumentParser()
parser.add_argument('-s','--species',
                    default = 'ce_hs',
                    type = str,
                    help = 'species')
parser.add_argument('-n','--network',
                    default = 'BioGRID',
                    type = str,
                    help = 'network')
parser.add_argument('-f','--features',
                    default = 'emb',
                    type = str,
                    help = 'features')
parser.add_argument('-i','--indices',
                    default = '0_10',
                    type = str,
                    help = 'indices of GSC terms to do')
args = parser.parse_args()
species = args.species
network = args.network
features = args.features
indices = args.indices
indices_int = [int(item) for item in indices.strip().split('_')]

if species.split("_")[0] in ["ce","dm","dr"]:
    other_spec = species.split("_")[0]
else:
    other_spec = species.split("_")[1]
    
def get_sim(coef_hs,coef_other):
    cos_sim = 1 - cosine(coef_hs,coef_other)
    return cos_sim
    
def get_info(akey,info_dict,w_json,label_dict,specie,net_arr,all_genes_in_network):
    if akey in info_dict[specie]:
        pass
    else:
        info_dict[specie][akey] = {}
        tic = time.time()
        info_dict[specie][akey]["coef"] = np.array(w_json[akey]['weights'])
        info_dict[specie][akey]["intercept"] = float(w_json[akey]['intercept'])
        Xtstind = []
        ytst = []
        tic = time.time()
        for agene in label_dict[akey]['Train-Positive']:
            Xtstind.append(np.where(all_genes_in_network==agene)[0][0])
            ytst.append(1)
        for agene in label_dict[akey]['Train-Negative']:
            Xtstind.append(np.where(all_genes_in_network==agene)[0][0])
            ytst.append(0)
        for agene in label_dict[akey]['Test-Positive']:
            Xtstind.append(np.where(all_genes_in_network==agene)[0][0])
            ytst.append(1)
        for agene in label_dict[akey]['Test-Negative']:
            Xtstind.append(np.where(all_genes_in_network==agene)[0][0])
            ytst.append(0)
        info_dict[specie][akey]["Xtst"] = np.array(Xtstind)
        info_dict[specie][akey]["ytst"] = np.array(ytst)
    return info_dict
    
def get_metric(info_dict,weights,weights_key,task,task_key,net_arr):
    # get that data
    Xtst = net_arr[info_dict[task][task_key]["Xtst"],:]
    ytst = info_dict[task][task_key]["ytst"]
    coef = info_dict[weights][weights_key]["coef"]
    intercept = info_dict[weights][weights_key]["intercept"]
    num_tst_pos = (ytst == 1).sum()
    # calculate things
    probs = expit(np.matmul(Xtst,coef.T) + intercept)
    avgp = average_precision_score(ytst,probs)
    prior = num_tst_pos/Xtst.shape[0]
    log2_prior = np.log2(avgp/prior)
    auroc = roc_auc_score(ytst,probs)
    return log2_prior, auroc

print("Loading weight jsons")
tic = time.time()
# load weight jsons
fp_weights = "../results/bothterms_weights/"
fn_hs = fp_weights + f"GO_hs__{species}__{network}__{features}.json"
with open(fn_hs) as f:
  w_hs = json.load(f)
print("Num hs terms",len(list(w_hs)))

fn_other = fp_weights + f"GO_{other_spec}__{species}__{network}__{features}.json"
with open(fn_other) as f:
  w_other = json.load(f)
print("Time to load jsons is", time.time()-tic)
print("Loading GSCs")
tic = time.time()
# load GSCs
fp_GSCs = "../data/GSCs/GSCsBothTerms/"
fn_gsc_hs = fp_GSCs + f"MainLabels__MethodNum-0__GO_hs__{species}__15_200_0.5_0.5__{network}.json"
with open(fn_gsc_hs) as f:
  gsc_hs = json.load(f)
fn_gsc_other = fp_GSCs + f"MainLabels__MethodNum-0__GO_{other_spec}__{species}__15_200_0.5_0.5__{network}.json"
with open(fn_gsc_other) as f:
  gsc_other = json.load(f)
print("The time it took to load the GSCs is",time.time()-tic)
print("Reading in the network data")
tic = time.time()
# This section will read in the network data
if network == "BioGRID":
    OTFtype = "Sparse"
elif network == "IMP":
    OTFtype = "Dense"
if features == 'adj':
    FN_adj = '../data/features/AdjMat_Influence/'
    FN_end = f"{species}__{network}_raw__eggnog_direct_AllOnes.adj"
    with open(FN_adj+FN_end+'.json') as f:
      net_dict = json.load(f)
    print('The number of nodes in the network is',len(net_dict))
    all_genes_in_network = np.array(list(net_dict))
    net_arr = np.zeros((len(net_dict),len(net_dict)))
elif features in ['emb',"allemb"]:
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
    net_arr = np.zeros((len(net_dict),len(emb_tmp)))
    print('The number of nodes in the network is',len(net_dict))
    print('The dimesions of the embeddings is',len(emb_tmp))
print("The time to took to read network data", time.time()-tic)


print("Making the network array")
tic = time.time()
all_genes_in_network = np.array(list(net_dict))
for idx, agene in enumerate(all_genes_in_network):
    net_arr[idx,:] = net_dict[agene]
print("The time it took to make net_arr is",time.time()-tic)


print("Making the metric arrays")
tic = time.time()
sims_arr = np.zeros((len(list(w_hs)[indices_int[0]:indices_int[1]]),len(list(w_other))))
log2p_arr_hs_to_other = np.zeros((len(list(w_hs)[indices_int[0]:indices_int[1]]),len(list(w_other))))
log2p_arr_other_to_hs = np.zeros((len(list(w_hs)[indices_int[0]:indices_int[1]]),len(list(w_other))))
auroc_arr_hs_to_other = np.zeros((len(list(w_hs)[indices_int[0]:indices_int[1]]),len(list(w_other))))
auroc_arr_other_to_hs = np.zeros((len(list(w_hs)[indices_int[0]:indices_int[1]]),len(list(w_other))))
info_dict = {}
info_dict["hs"] = {}
info_dict[other_spec] = {}
for idx1, key_hs in enumerate(list(w_hs)[indices_int[0]:indices_int[1]]):
    info_dict = get_info(key_hs,info_dict,w_hs,gsc_hs,"hs",net_arr,all_genes_in_network)
    for idx2, key_other in enumerate(list(w_other)):
        tic = time.time()
        info_dict = get_info(key_other,info_dict,w_other,gsc_other,other_spec,net_arr,all_genes_in_network)
        tic = time.time()
        cos_sim = get_sim(info_dict['hs'][key_hs]["coef"],info_dict[other_spec][key_other]["coef"])
        sims_arr[idx1,idx2] = cos_sim
        tic = time.time()
        log2_prior, auroc = get_metric(info_dict,"hs",key_hs,other_spec,key_other,net_arr)
        log2p_arr_hs_to_other[idx1,idx2] = log2_prior
        auroc_arr_hs_to_other[idx1,idx2] = auroc
        tic = time.time()
        log2_prior, auroc = get_metric(info_dict,other_spec,key_other,"hs",key_hs,net_arr)
        log2p_arr_other_to_hs[idx1,idx2] = log2_prior
        auroc_arr_other_to_hs[idx1,idx2] = auroc
print("The time it took to make the metric arrays is",time.time()-tic)


fp_weights = "../results/bothterms_weights/"
np.save(fp_weights+f"CosSims__GO__{species}__{network}__{features}__{indices}.npy",sims_arr)
np.save(fp_weights+f"Log2P_hs_to_{other_spec}__GO__{species}__{network}__{features}__{indices}.npy",log2p_arr_hs_to_other)
np.save(fp_weights+f"Log2P_{other_spec}_to_hs__GO__{species}__{network}__{features}__{indices}.npy",log2p_arr_other_to_hs)
np.save(fp_weights+f"auROC_hs_to_{other_spec}__GO__{species}__{network}__{features}__{indices}.npy",auroc_arr_hs_to_other)
np.save(fp_weights+f"auROC_{other_spec}_to_hs__GO__{species}__{network}__{features}__{indices}.npy",auroc_arr_other_to_hs)
