import json
import glob
import pandas as pd
import numpy as np
from scipy.stats import zscore, norm
import utls
from statsmodels.stats.multitest import fdrcorrection
import argparse


'''
This script will read in gene predictions from a model trained on a known
disease gene set and rank the gene preidction and find the most relevant
biological processes and phenotypes for each species.

Note: Some code is commented out that adds gene symbol and gene name info to
the saved dataframes.
'''

### add some conversion dictionaries
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


parser = argparse.ArgumentParser()
parser.add_argument("-n","--network",
                    default = "IMP",
                    type = str,
                    help = "type of network to do")
parser.add_argument("-f","--features",
                    default = "ce_dm_dr_hs_mm_sc",
                    type = str,
                    help = "features used in dis model")
parser.add_argument("-di","--disind",
                    default = 0,
                    type = int,
                    help = "which disease to select")
parser.add_argument("-t","--task",
                    default = "Gene",
                    type = str,
                    help = "either GO, Monarch or Gene")
parser.add_argument("-s","--species",
                    default = "hs",
                    type = str,
                    help = "species to look at")
args = parser.parse_args()

network = args.network
features = args.features
dis_ind = args.disind
task = args.task
species = args.species
fpsave = "../results/disease_evals/"

print("The network is",network)
print("The features are",features)


# get the top disease models
dfmet = pd.read_csv(fpsave+f"DisMet__{features}__{network}.tsv",sep="\t")

#select a certain disease
ID = dfmet.iloc[dis_ind]["ID"]
name = dfmet.iloc[dis_ind]["name"]
ID2 = ID.replace(":","")
print("The disease being done is",ID, name)

# get the preds for the disease
fppred = "../results/disease_preds/"
FNpred = fppred + f"*{features}*{network}*{ID}*.tsv"
FNpred = glob.glob(FNpred)[0]
dfpred = pd.read_csv(FNpred,sep="\t")
dfpred["Gene"] = dfpred["Gene"].astype(str)
dfpred = dfpred.sort_values(by=["Gene"],ascending=True)

# get disease postiives
fpgsc_dis= "../data/GSCs/GSCs/"
FNdis = glob.glob(fpgsc_dis+f"*-0__DisNet*{network}.json")[0]
with open(FNdis) as f:
    dis_json = json.load(f)
dis_pos = np.array(dis_json[ID]["Train-Positive"]+dis_json[ID]["Test-Positive"])
dis_neg = np.array(dis_json[ID]["Train-Negative"]+dis_json[ID]["Test-Negative"])

# ortho information
if species != "hs":
    dfortho = utls.load_ortho_pairs(network,species)
    ortho_pos = dfortho[dfortho["hs"].isin(dis_pos)][species].unique()
    ortho_neg = dfortho[dfortho["hs"].isin(dis_neg)][species].unique()

# get genes in network
all_genes_in_network = utls.get_genes_from_network(network,species)


if task in ["GO","Monarch"]:
    fp_gsc_dis = "../data/GSCs/GSCDis/"
    with open(fp_gsc_dis + f"DisGSCs__{task}__{species}__{network}.json") as handle:
        data_dict = json.load(handle)
    dfgenes = dfpred[dfpred["Gene"].isin(all_genes_in_network)]
    all_mean = np.mean(dfgenes["Prob"].to_numpy())
    all_std = np.std(dfgenes["Prob"].to_numpy())
    if species != "hs":
        ortho_pos = dfortho[dfortho["hs"].isin(dis_pos)][species].unique()
        ortho_neg = dfortho[dfortho["hs"].isin(dis_neg)][species].unique()
    # generate top GSC prediction table
    results = [] 
    for idx, akey in enumerate(data_dict):
        term = data_dict[akey]["Name"]
        genes = data_dict[akey]["Genes"]
        set_probs = dfgenes[dfgenes["Gene"].isin(genes)]["Prob"].to_numpy()
        set_mean = np.mean(set_probs)
        set_len = len(set_probs)
        set_z = (set_mean-all_mean) / (all_std/np.sqrt(set_len))
        # find ortho log info
        if species != "hs":
            sp_orthos = np.intersect1d(genes,dfortho[species].unique())
            num_ortho = len(sp_orthos)
            num_ortho_pos = len(np.intersect1d(genes,ortho_pos))
            num_ortho_neg = len(np.intersect1d(genes,ortho_neg))
            results.append([akey,term,len(genes),num_ortho,
                            num_ortho_pos,num_ortho_neg,set_z])
            col_names = ["ID","Name","NumGenes","NumOrthoHuman",
                         "NumOrthoDisPos","NumOrthoDisNeg","ProbsZScore"]
        else:
            num_pos_in_dis_pos = len(np.intersect1d(genes,dis_pos))
            num_pos_in_dis_neg = len(np.intersect1d(genes,dis_neg))
            results.append([akey,term,len(genes),num_pos_in_dis_pos,
                            num_pos_in_dis_neg,set_z])
            col_names = ["ID","Name","NumPos","NumGeneDisPos",
                         "NumGeneDisNeg","ProbsZScore"]

    dfresults = pd.DataFrame(results,columns=col_names)
    # make some other metrics
    zscores = dfresults["ProbsZScore"].to_numpy()
    pvals = norm.sf(abs(zscores))
    a, fdrs = fdrcorrection(pvals)
    dfresults["Pval"] = pvals
    dfresults["FDR"] = fdrs
    dfresults = dfresults.sort_values(by=["ProbsZScore"],ascending=False)
    print("The number of terms in the GSC is",dfresults.shape[0])
    dfresults.to_csv(fpsave +f"{dis_ind}__{ID2}__{task}__{species}__{network}__{features}.tsv",
              sep="\t",header=True,index=False)
elif task == "Gene":
    dfgenes = dfpred[dfpred["Gene"].isin(all_genes_in_network)]
    dfgenes = dfgenes.sort_values(by=["Prob"],ascending=False)
    dfgenes = dfgenes
    zscores = zscore(dfgenes["Prob"].to_numpy())
    pvals = norm.sf(abs(zscores))
    a, fdrs = fdrcorrection(pvals)
    dfgenes["Zscore"] = zscores
    dfgenes["Pval"] = pvals
    dfgenes["FDR"] = fdrs
    if species != "hs":
        dfpos = dfortho[dfortho["hs"].isin(dis_pos)]
        dfneg = dfortho[dfortho["hs"].isin(dis_neg)]
    fpMGI = "../data/MGI/"
    fpMGI = fpMGI + f"{names_MGI_dict[species]}/"
    symbols = []
    names = []
    isortho = []
    ortho_pos = []
    ortho_neg = []
    indis = []
    for idx, agene in enumerate(dfgenes["Gene"].to_list()):
#         try:
#            with open(fpMGI+f"entrezgeneID-{agene}.json") as f:
#                mgi = json.load(f)
#            symbols.append(mgi["symbol"])
#            names.append(mgi["name"])
#        except FileNotFoundError:
#            symbols.append("")
#            names.append("")
        if species != "hs":
            if agene in dfortho[species].to_list():
                isortho.append("yes")
            else:
                isortho.append("no")
            if dfpos[dfpos[species]==agene].shape[0] > 0:
                ortho_pos.append("yes")
            else:
                ortho_pos.append("no")
            if dfneg[dfneg[species]==agene].shape[0] > 0:
                ortho_neg.append("yes")
            else:
                ortho_neg.append("no")
        else:
            if agene in dis_pos:
                indis.append("Pos")
            elif agene in dis_neg:
                indis.append("Neg")
            else:
                indis.append("Neither")
#     dfgenes["Symbol"] = symbols
#     dfgenes["Name"] = names
    if species != "hs":
        dfgenes["Ortho?"] = isortho
        dfgenes["OrthoPos?"] = ortho_pos
        dfgenes["OrthoNeg?"] = ortho_neg
    else:
        dfgenes["InDis?"] = indis
    dfgenes.to_csv(fpsave +f"{dis_ind}__{ID2}__{task}__{species}__{network}__{features}.tsv",
              sep="\t",header=True,index=False)
