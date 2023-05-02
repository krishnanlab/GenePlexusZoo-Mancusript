# zoo_tmp
These scripts are what were used to generate the findings in *Joint representation of molecular networks from multiple species improves gene classification*. The data generated for this manuscript was very large and computationally expensive to generate. These scripts provide the main code that was able to generate the results, however we don't provide computational support or HPC job scheduling code to reproduce the data in full.

## Package Versions

The code was tested with `python==3.8.3`.

`requirements.txt` contains packages used for all scripts except those that use `pecanpy`, whose pacakges can be found in `requirements_pecanpy.txt`.

### Data
The data used in this study is available on [Zenodo](https://zenodo.org/record/3352348/). To get the data run
```
sh get_data.sh
```

## File Structure

In the `src` folder there is : 

`01`  

These scripts deal with directly processing the edgelists into downstream data.

`02`

These scripts make the feature matrices used in the machine learning models.  

`03`

These scripts generate the main findings (Fig 2 and Fig 3).  

`04`

These scripts generate results from the matched GSC analyses.  

`05`

These scripts generate results for training human disease models and looking at predictions across species.  

`figure_code`

This generates the figures contained in the manuscript.  
