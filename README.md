# NISC
Neural Network-based Imputation (NISC) for Single Cell RNA-seq count data

NISC uses autoencoder, a learning technique, with a weighted loss function and regularization to impute the dropouts in scRNA-seq count data.

What we need:

* Python 3.6.4
* TensorFlow 1.15.0
* NumPy 1.16.1

In this folder, there are two execution python code files with data:

1. The file "nisc_impute.py".

     It will read raw count data "example_raw_data.npy" and conduct the imputation. The example dataset contains 1000 cells (in columns) and 800 genes (in rows). The sparsity of the raw data is about 70%. The output file "example_nisc_imputation.npy" stores the result after imputation.

2. The file "generate_plots.py".

      It will generate tSNE plots, UMAP, and PCA plots for ground truth (no extra zero added), noisy input data, and imputed data after NISC.

  - "cellType.npy" stores the information of cell types and there are two cell types for this example dataset.
  - "Truth.npy" stores the ground truth values of gene expressions.
