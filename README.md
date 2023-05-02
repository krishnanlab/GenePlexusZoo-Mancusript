# GenePlexusZoo – Manuscript
This repository contains the scripts used to generate the findings in the manuscript *Joint representation of molecular networks from multiple species improves gene classification*. The data for this manuscript were very large and computationally expensive to generate. While the scripts here provide the main code for generating the results, reproducing the data in full requires cluster computating resources and HPC job scheduling code, which are not provided here.

## Package versions

The code was tested with `python==3.8.3`.

`requirements.txt` contains packages used for all scripts except those that use `pecanpy`, whose pacakges can be found in `requirements_pecanpy.txt`.

### Data
The data used in this study is available on [Zenodo](the data available at https://zenodo.org/record/7888044). To get the data run
```
sh get_data.sh
```

## File Structure

The `src` folder contains the following subfolders with scripts for different stages of the analysis: 

* `01`: Processing the edgelists into downstream data.

* `02`: Making the feature matrices used in the machine learning models.  

* `03`: Generating the main findings (Fig 2 and Fig 3).  

* `04`: Generating results from the matched geneset collection (GSC) analyses.  

* `05`: Generating results for training human disease models and looking at predictions across species. 

The `figure_code` folder contains the scripts to generate the figures in the manuscript.  
