# Cluster Analysis

This repo contains notebooks and functions used to run cluster analysis on biomass products. They can be clustered by either location or products.

## Files

The repository includes the following files and directories:

- `By_Prod\` contains files used in an intial clustering attempt by locations
- `By_Loc\` contains files used in an intial clustering attempt by products
- `clustering_explorations` contains notebooks with clustering results for various locations, as well as a Simulation exercise
- `data` contains all the datasets used for the clustering 
- `Prod_Clustering_Template.ipynb`: template with demonstration of how to run and visualize the clustering
- `Clusters_by_n.ipynb`: exploration using different parameters in clustering algorithms to find appropriate values
- `similarity_metric_template.ipynb` : template to find product similarity using hamming distance
- `utils.py` : contains functions used during clustering and analysis and preparation of data
- `utils_char_analysis.py` : contains functions used for finding product similarity metrics
- `README.md`: This file, providing an overview of the repository and instructions for running the code.

## Dependencies

The code requires the following dependencies:
- Python 3.x
- NumPy
- python sklearn
- Pandas
- seaborn
- matplotlib
- scipy
- import functions from utils/utils_char_analysis
  
