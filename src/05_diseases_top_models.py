import glob
import pandas as pd


'''
This script looks through the evaluation metrics for disease models trained
using 03_main_get_matrics.py and sorts the diseases by log2(auPRC/prior)
'''

features = "ce_hs"
fpsave = "../results/disease_evals/"

for network in ["BioGRID","IMP"]:
    # get the top disease models
    fpmet = "../results/metrics/"
    FNmet = fpmet + f"*__DisNet_hs*{features}__{network}*SummedDegree-0.50*2000*0.1_0.001*.tsv"
    FNmet = glob.glob(FNmet)[0]
    dfmet = pd.read_csv(FNmet,sep="\t").sort_values(by=["log2p"],ascending=False)
    dfmet = dfmet[["ID","name","log2p","num_train_pos","num_test_pos"]]
    dfmet.to_csv(fpsave+f"DisMet__{features}__{network}.tsv",sep="\t",header=True,index=False)

