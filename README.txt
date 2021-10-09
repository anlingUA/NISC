NISC: Neural Network-Imputation for Single-Cell RNA Sequencing Data on Clustering Analysis

Introduction: We develop a Neural Network-based Imputation for scRNA-seq count data, NISC. It uses autoencoder, a learning technique, with a weighted loss function and regularization to impute the dropouts in scRNA-seq count data. A systematic evaluation shows that NISC is an effective imputation approach for sparse scRNA-seq count data and its performance surpasses existing imputation methods in cell type identification. 

* Python 3.6.4
* TensorFlow 1.15.0
* NumPy 1.16.1

In this folder, there are two execution python code files with data:

1.The file "nisc_impute.py".
It will read sparse raw data "example_raw_data.npy", and conduct the imputation. The raw data has 1000 cells and 800 genes. Each row represents a gene, each column represents a cell. The sparsity of the raw data is 70%.
It will generate "example_nisc_imputation.npy", which stores the result after imputation.

2.The file "generate_plots.py".
It will generate tSNE plots and PCA plots for ground truth, with dropout and NISC imputation.

"cellType.npy" stores the information of cell types for this simulated data. There are two cell types in this simulated data.

"Truth.npy" stores the ground truth values of gene expressions in this data.

