species=ce_dr
network=BioGRID_raw
weights=AllOnes
input=../data/edgelists/full/${species}__${network}__direct__${weights}.edgelist
dims=500
mode=SparseOTF
p=0.1
q=0.1
walklen=120
workers=8
window=10
edges=weighted
direct=undirected
output=../data/features/embeddings/${species}__${network}__eggnog_direct_${weights}__Pecanpy_${dims}_${mode}_${p}_${q}_${walklen}_${workers}_${window}_${edges}_${direct}.emb
echo $output

python edgelists_write_full_edgelist.py -i ${species}__${network}__direct__${weights}
pecanpy --input $input --output $output --dimensions $dims --mode $mode --p $p --q $q --walk-length $walklen --workers $workers --window-size $window --$edges --$direct
