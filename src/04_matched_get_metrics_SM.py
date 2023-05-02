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
import copy


'''
This script is used to generate the results used in the supplemental material
regarding the matched gene set evaluations

PosNeg NoOrtho and single are used in the anlaysis combining genes from two 
species during training

cross is used when evaluating training in one modle and predicting in another
'''


parser = argparse.ArgumentParser()
parser.add_argument('-t','--task',
                    default = 'GO',
                    type = str,
                    help = 'task to do')
parser.add_argument('-g','--geneset_info',
                    default = 'hs_mm__15_200_0.5_0.5',
                    type = str,
                    help = 'information needed to make genesets')
parser.add_argument('-n','--network_info',
                    default = 'mm__BioGRID_raw__eggnog_direct_AllOnes__Pecanpy_2000_SparseOTF_0.1_0.001_120_8_10_weighted_undirected.emb',
                    type = str,
                    help = 'information needed to get feature matrix')
parser.add_argument('-m','--model_info',
                    default = 'NoScaling__LR_l2_1.0',
                    type = str,
                    help = 'information on model and how to scale the data')
parser.add_argument('-a','--add_genes',
                    default="single",
                    type=str,
                    help='how to combine genes [PosNeg, Pos, PosNoOrtho, PosNegNoOrtho, single, cross]')
args = parser.parse_args()
task = args.task
geneset_info = args.geneset_info
network_info = args.network_info
model_info = args.model_info
add_genes = args.add_genes
if add_genes == "single":
    network = network_info.split('__')[0].split('_')[0]
else:
    network = network_info.split('__')[1].split('_')[0]
sp0 = geneset_info.split('__')[0].split('_')[0]
sp1 = geneset_info.split('__')[0].split('_')[1]

print('The task is',task)
print('The geneset_info is',geneset_info)
print('The network_info is',network_info)
print('The model_info is',model_info)

tic0 = time.time()
tic1 = time.time()
# This section will read in the network data
print('Reading in network data')
if '.adj' in network_info:
    if add_genes != "single":
        FN_adj = '../data/features/AdjMat_Influence/'
        with open(FN_adj+network_info+'.json') as f:
            net_dict = json.load(f)
        print('The number of nodes in the network is',len(net_dict))
        all_genes_in_network = np.array(list(net_dict))
    else:
        single_net_dicts = {}
        for aspec in [sp0,sp1]:
            FN_adj = '../data/features/AdjMat_Influence/'
            with open(FN_adj+f"{aspec}__"+network_info+'.json') as f:
                net_dict = json.load(f)
            print('The number of nodes in the network is',len(net_dict))
            all_genes_in_network = np.array(list(net_dict))
            single_net_dicts[aspec] = net_dict

elif '.emb' in network_info:
    if add_genes != "single":
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
    else:
        single_net_dicts = {}
        for aspec in [sp0,sp1]:
            FN_net = '../data/features/embeddings/'+f"{aspec}__"+'%s'%network_info
            net_dict = {}
            with open(FN_net,'r') as f:
                for idx, line in enumerate(f):
                    if idx == 0:
                        continue
                    line = line.strip().split()
                    gene_tmp = line[0]
                    emb_tmp = [float(item) for item in line[1:]]
                    net_dict[gene_tmp] = emb_tmp
            single_net_dicts[aspec] = net_dict


        print('The number of nodes in the network is',len(net_dict))
        print('The dimesions of the embeddings is',len(emb_tmp))
print('The time in  minutes it took to read in network data is',int((time.time()-tic1)/60))

if add_genes not in ["cross"]:
    # Get ortholog info
    fp_ortho = "../data/edgelists/connections/"
    sp0_orthos = {}
    sp1_orthos = {}
    with open(fp_ortho+f"{sp0}_{sp1}__{network}_raw__direct__AllOnes.edgelist","r") as f:
        for line in f:
            gene0_tmp = line.split()[0]
            gene1_tmp = line.split()[1]
            if gene0_tmp != sp0_orthos:
                sp0_orthos[gene0_tmp] = [gene1_tmp]
            else:
                sp0_orthos[gene0_tmp] = sp0_orthos[gene0_tmp] + [gene1_tmp]
            if gene1_tmp != sp1_orthos:
                sp1_orthos[gene1_tmp] = [gene0_tmp]
            else:
                sp1_orthos[gene1_tmp] = sp1_orthos[gene1_tmp] + [gene0_tmp]

tic2 = time.time()
# This section will load the label_dict
fp_GSCs = '../data/GSCs/GSCsBothTerms/'
with open(fp_GSCs+'MainLabels__MethodNum-0__%s_%s__%s__%s.json'%(task,sp0,geneset_info,network)) as f:
    label_dict_sp0 = json.load(f)
label_dict_Bothsp0 = copy.deepcopy(label_dict_sp0)
print('The number of genesets in the label_dict is',len(label_dict_sp0))
with open(fp_GSCs+'MainLabels__MethodNum-0__%s_%s__%s__%s.json'%(task,sp1,geneset_info,network)) as f:
    label_dict_sp1 = json.load(f)
label_dict_Bothsp1 = copy.deepcopy(label_dict_sp1)
print('The number of genesets in the label_dict is',len(label_dict_sp1))
print('The time in  minutes it took to read in label data is',int((time.time()-tic2)/60))

if add_genes == "PosNeg":
    for anID in label_dict_sp0:
        for aset in ["Train-Positive", "Train-Negative"]:
            label_dict_Bothsp0[anID][aset] = label_dict_sp0[anID][aset]+label_dict_sp1[anID][aset]
            label_dict_Bothsp1[anID][aset] = label_dict_sp0[anID][aset]+label_dict_sp1[anID][aset]
        for agene in label_dict_sp1[anID]["Train-Positive"]:
            if agene in sp1_orthos:
                label_dict_Bothsp0[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp0[anID]["Train-Negative"],
                                                                               sp1_orthos[agene]))
                label_dict_Bothsp1[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp1[anID]["Train-Negative"],
                                                                               sp1_orthos[agene]))
        for agene in label_dict_sp0[anID]["Train-Positive"]:
            if agene in sp0_orthos:
                label_dict_Bothsp0[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp0[anID]["Train-Negative"],
                                                                               sp0_orthos[agene]))
                label_dict_Bothsp1[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp1[anID]["Train-Negative"],
                                                                               sp0_orthos[agene]))
elif add_genes == "Pos":
    for anID in label_dict_sp0:
        label_dict_Bothsp0[anID]["Train-Positive"] = label_dict_sp0[anID]["Train-Positive"]+label_dict_sp1[anID]["Train-Positive"]
        label_dict_Bothsp1[anID]["Train-Positive"] = label_dict_sp0[anID]["Train-Positive"]+label_dict_sp1[anID]["Train-Positive"]
        for agene in label_dict_sp1[anID]["Train-Positive"]:
            if agene in sp1_orthos:
                label_dict_Bothsp0[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp0[anID]["Train-Negative"],
                                                                               sp1_orthos[agene]))
        for agene in label_dict_sp0[anID]["Train-Positive"]:
            if agene in sp0_orthos:
                label_dict_Bothsp1[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp1[anID]["Train-Negative"],
                                                                               sp0_orthos[agene]))
elif add_genes == "PosNoOrtho":
    for anID in label_dict_sp0:
        for agene in label_dict_sp1[anID]["Train-Positive"]:
            if agene in sp1_orthos:
                label_dict_Bothsp0[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp0[anID]["Train-Negative"],
                                                                               sp1_orthos[agene]))
                if len(np.intersect1d(sp1_orthos[agene],label_dict_Bothsp0[anID]["Train-Positive"])) == 0:
                    label_dict_Bothsp0[anID]["Train-Positive"] = label_dict_Bothsp0[anID]["Train-Positive"] + [agene]
            else:
                label_dict_Bothsp0[anID]["Train-Positive"] = label_dict_Bothsp0[anID]["Train-Positive"] + [agene]
        for agene in label_dict_sp0[anID]["Train-Positive"]:
            if agene in sp0_orthos:
                label_dict_Bothsp1[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp1[anID]["Train-Negative"],
                                                                               sp0_orthos[agene]))
                if len(np.intersect1d(sp0_orthos[agene],label_dict_Bothsp1[anID]["Train-Positive"])) == 0:
                    label_dict_Bothsp1[anID]["Train-Positive"] = label_dict_Bothsp1[anID]["Train-Positive"] + [agene]
            else:
                label_dict_Bothsp1[anID]["Train-Positive"] = label_dict_Bothsp1[anID]["Train-Positive"] + [agene]

elif add_genes == "PosNegNoOrtho":
    for anID in label_dict_sp0:
        # do postives
        for agene in label_dict_sp1[anID]["Train-Positive"]:
            if agene in sp1_orthos:
                if len(np.intersect1d(sp1_orthos[agene],label_dict_Bothsp0[anID]["Train-Positive"])) == 0:
                    label_dict_Bothsp0[anID]["Train-Positive"] = label_dict_Bothsp0[anID]["Train-Positive"] + [agene]
            else:
                label_dict_Bothsp0[anID]["Train-Positive"] = label_dict_Bothsp0[anID]["Train-Positive"] + [agene]
        for agene in label_dict_sp0[anID]["Train-Positive"]:
            if agene in sp0_orthos:
                if len(np.intersect1d(sp0_orthos[agene],label_dict_Bothsp1[anID]["Train-Positive"])) == 0:
                    label_dict_Bothsp1[anID]["Train-Positive"] = label_dict_Bothsp1[anID]["Train-Positive"] + [agene]
            else:
                label_dict_Bothsp1[anID]["Train-Positive"] = label_dict_Bothsp1[anID]["Train-Positive"] + [agene]
        # do negatives
        label_dict_Bothsp0[anID]["Train-Negative"] = label_dict_sp0[anID]["Train-Negative"]+label_dict_sp1[anID]["Train-Negative"]
        label_dict_Bothsp1[anID]["Train-Negative"] = label_dict_sp0[anID]["Train-Negative"]+label_dict_sp1[anID]["Train-Negative"]
        for agene in label_dict_sp1[anID]["Train-Positive"]:
            if agene in sp1_orthos:
                label_dict_Bothsp0[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp0[anID]["Train-Negative"],
                                                                               sp1_orthos[agene]))
                label_dict_Bothsp1[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp1[anID]["Train-Negative"],
                                                                               sp1_orthos[agene]))
        for agene in label_dict_sp0[anID]["Train-Positive"]:
            if agene in sp0_orthos:
                label_dict_Bothsp0[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp0[anID]["Train-Negative"],
                                                                               sp0_orthos[agene]))
                label_dict_Bothsp1[anID]["Train-Negative"] = list(np.setdiff1d(label_dict_Bothsp1[anID]["Train-Negative"],
                                                                               sp0_orthos[agene]))

elif add_genes == "single":
    for anID in label_dict_sp0:
        for agene in label_dict_sp1[anID]["Train-Positive"]:
            if agene in sp1_orthos:
                if len(np.intersect1d(sp1_orthos[agene],label_dict_Bothsp0[anID]["Train-Positive"])) == 0:
                    label_dict_Bothsp0[anID]["Train-Positive"] = list(set(label_dict_Bothsp0[anID]["Train-Positive"] + sp1_orthos[agene]))
        for agene in label_dict_sp0[anID]["Train-Positive"]:
            if agene in sp0_orthos:
                if len(np.intersect1d(sp0_orthos[agene],label_dict_Bothsp1[anID]["Train-Positive"])) == 0:
                    label_dict_Bothsp1[anID]["Train-Positive"] = list(set(label_dict_Bothsp1[anID]["Train-Positive"] + sp0_orthos[agene]))

elif add_genes == "cross":
    for anID in label_dict_sp0:
        label_dict_Bothsp0[anID]["Test-Positive"] = label_dict_sp1[anID]["Test-Positive"]
        label_dict_Bothsp0[anID]["Test-Negative"] = label_dict_sp1[anID]["Test-Negative"]
        label_dict_Bothsp1[anID]["Test-Positive"] = label_dict_sp0[anID]["Test-Positive"]
        label_dict_Bothsp1[anID]["Test-Negative"] = label_dict_sp0[anID]["Test-Negative"]

tic3 = time.time()
print('Doing ML part')
if "single" in add_genes:
    mymethods = [(label_dict_sp0,'%s-Trn-%s-Tst-%s'%(add_genes,sp0,sp0)),
                 (label_dict_sp1,'%s-Trn-%s-Tst-%s'%(add_genes,sp1,sp1)),
                 (label_dict_Bothsp0,'%s-Trn-%s_%s-Tst-%s'%(add_genes,sp0,sp1,sp0)),
                 (label_dict_Bothsp1,'%s-Trn-%s_%s-Tst-%s'%(add_genes,sp0,sp1,sp1))]

elif "cross" == add_genes:
    mymethods = [(label_dict_sp0,'%s-Trn-%s-Tst-%s'%(add_genes,sp0,sp0)),
                 (label_dict_sp1,'%s-Trn-%s-Tst-%s'%(add_genes,sp1,sp1)),
                 (label_dict_Bothsp0,'%s-Trn-%s-Tst-%s'%(add_genes,sp0,sp1)),
                 (label_dict_Bothsp1,'%s-Trn-%s-Tst-%s'%(add_genes,sp1,sp0))]

else:
    mymethods = [(label_dict_sp0,'%s-Trn-%s-Tst-%s'%(add_genes,sp0,sp0)),
                 (label_dict_sp1,'%s-Trn-%s-Tst-%s'%(add_genes,sp1,sp1)),
                 (label_dict_Bothsp0,'%s-Trn-%s_%s-Tst-%s'%(add_genes,sp0,sp1,sp0)),
                 (label_dict_Bothsp1,'%s-Trn-%s_%s-Tst-%s'%(add_genes,sp0,sp1,sp1))]
for label_dict, method in mymethods:
    if add_genes == "single":
        net_dict = single_net_dicts[method.split("-")[-1]]
    results_tmp = []
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
        Xtst = []
        ytst = []
        for agene in label_dict[akey]['Test-Positive']:
            Xtst.append(net_dict[agene])
            ytst.append(1)
        for agene in label_dict[akey]['Test-Negative']:
            Xtst.append(net_dict[agene])
            ytst.append(0)
        Xtrn = np.array(Xtrn)
        Xtst = np.array(Xtst)
        ytrn = np.array(ytrn)
        ytst = np.array(ytst)
        num_trn_pos = len(label_dict[akey]['Train-Positive'])
        num_tst_pos = len(label_dict[akey]['Test-Positive'])
        # scale the data if desired
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
        # train and predict
        clf.fit(Xtrn,ytrn)
        probs = clf.predict_proba(Xtst)[:,1]
        # evaluate the model
        avgp = average_precision_score(ytst,probs)
        prior = num_tst_pos/Xtst.shape[0]
        log2_prior = np.log2(avgp/prior)
        auroc = roc_auc_score(ytst,probs)
        probs_sorted = np.argsort(probs)[::-1]
        nhits = (ytst[probs_sorted[0:num_tst_pos]] > 0).sum()
        # save results
        if add_genes == "single":
            myspec = method.split("-")[-1]
            network_info = f"{myspec}__" + args.network_info

        net_combo = network_info.split('__')
        if '.emb' in network_info:
            net_combo = network_info.split('.em')[0].split('__')
        elif '.adj' in network_info:
            net_combo = network_info.split('.ad')[0].split('__')+['adj']

        results_tmp.append([geneset_info,method,
                            net_combo[1],net_combo[0],net_combo[2],net_combo[3], model_info,
                            akey,label_dict[akey]['Name'],num_trn_pos,Xtrn.shape[0],
                            num_tst_pos,Xtst.shape[0],
                            avgp,log2_prior,auroc,nhits])

    col_names = ['GSC_info','method',
                 'network','species','connections','feature_type', 'model_info',
                 'ID','name','num_train_pos','num_train_all',
                 'num_test_pos','num_test_all',
                 'avgp','log2p','auroc','PTopK']
    df_results = pd.DataFrame(results_tmp,columns=col_names)
    save_dir = '../results/bothterms_metrics_SM/'
    df_results.to_csv(save_dir + 'MetricResults--BothTerms--%s--%s--%s--%s--%s.tsv'%(task,method,geneset_info,network_info,model_info),
                      sep='\t',header=True,index=False)
print('The time in  minutes it took to do ML is',int((time.time()-tic3)/60))
print('The time in  minutes it took to do the script is',int((time.time()-tic0)/60))
