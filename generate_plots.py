import os
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import scipy.stats
import sklearn.manifold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import umap


def tsne_figure(data,perplexity=30,cluster_n=8,title="default"):

    cluster=np.load('cellType.npy')
    cluster=np.int32(cluster)
    data_t=np.transpose(data)
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=2000)
    tsne_results=tsne.fit_transform(data_t)
    kmeans = KMeans(init='k-means++', n_clusters=cluster_n, n_init=10)
    kmeans.fit(tsne_results)
    y_kmeans = kmeans.predict(tsne_results)


    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster, s=4, cmap='Set1')
    plt.title(title)
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    
def umap_figure(data,perplexity=30,cluster_n=8,title="default"):

    cluster=np.load('cellType.npy')
    cluster=np.int32(cluster)
    data_t=np.transpose(data)
    
    embedding = umap.UMAP(n_components=2, verbose=1)
    umap_results=embedding.fit_transform(data_t)
    kmeans = KMeans(init='k-means++', n_clusters=cluster_n, n_init=10)
    kmeans.fit(umap_results)
    y_kmeans = kmeans.predict(umap_results)


    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=cluster, s=4, cmap='Set1')
    plt.title(title)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')

def pca_figure_all(data,pca_n=2,cluster_n=4,title='default'):
    cluster=np.load('cellType.npy')
    cluster=np.int32(cluster)
    data_t=np.transpose(data)
    pca = PCA(n_components=pca_n)
    pca.fit(data_t)
    data_pca = pca.transform(data_t)
    kmeans = KMeans(init='k-means++', n_clusters=cluster_n, n_init=10)
    kmeans.fit(data_pca)
    y_kmeans = kmeans.predict(data_pca)

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster, s=4, cmap='Set1')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')


def pca_figure_selected(data,pca_n=2,cluster_n=4,title='default'):
    cluster=np.load('cellType.npy')
    cluster=np.int32(cluster)
    data_t=np.transpose(data)
    index=(cluster==1) | (cluster==5)


    pca = PCA(n_components=pca_n)
    pca.fit(data_t)
    data_pca = pca.transform(data_t)
    kmeans = KMeans(init='k-means++', n_clusters=cluster_n, n_init=10)
    kmeans.fit(data_pca)
    y_kmeans = kmeans.predict(data_pca)

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster, s=1, cmap='Set1')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
   
def draw_mean_var_figure(data):
    # compute the standard variance of the data in log scale 
    data_abs_var=np.std(np.log(data+1),axis=1)

    # compute the mean of the data in log scale 
    data_abs_mean=np.mean(np.log(data+1),axis=1)

    data_abs_cv=data_abs_var/data_abs_mean

    # find out the zero percentage of the data
    data_abs_non_zero=np.count_nonzero(data,axis=1)
    data_abs_zero=np.ones(data_abs_non_zero.shape)*data.shape[1]-data_abs_non_zero
    data_abs_zero_percent=data_abs_zero/data.shape[1]


    plt.figure()
    plt.subplot(2,2,1)
    plt.scatter(data_abs_var,data_abs_mean,c=data_abs_zero_percent, s=.1, cmap='viridis') #plasma, hsv
    cbar = plt.colorbar()
    cbar.set_label('zero percentage')
    plt.xlabel('std var')
    plt.ylabel('mean')


    plt.subplot(2,2,2)
    plt.scatter(data_abs_cv,data_abs_mean,c=data_abs_zero_percent, s=.1, cmap='viridis') #plasma, hsv
    cbar = plt.colorbar()
    cbar.set_label('zero percentage')
    plt.xlabel('cross variance')
    plt.ylabel('mean')
    plt.title('Total gene count: {0}. Non-zero gene count: {1}'.format(data.shape[0],np.sum(data_abs_zero_percent==0)))

    plt.subplot(2,2,3)
    plt.hist(data_abs_zero_percent,bins='auto')
    plt.xlabel('zero percentage')
    plt.ylabel('frequency')
    
    plt.subplot(2,2,4)
    plt.scatter(data_abs_zero_percent,data_abs_mean,c=data_abs_cv, s=.1, cmap='viridis') #plasma, hsv
    cbar = plt.colorbar()
    cbar.set_label('cross variance')
    plt.xlabel('zero percentage')
    plt.ylabel('mean')

    
    return data_abs_var, data_abs_mean, data_abs_zero_percent, data_abs_cv

    
data_count=np.load('example_raw_data.npy')


#code to compensate
data=data_count
gene_num=data.shape[0]

nn_output=np.load('example_nisc_imputation.npy')

print('nn output generated')


data_gt=np.transpose(np.load("Truth.npy"))
print('GT generated')



f=plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
pca_figure_all(np.log(data_gt+1),pca_n=2,cluster_n=2,title='Ground Truth')
plt.subplot(1,3,2)
pca_figure_all(np.log(data_count+1),pca_n=2,cluster_n=2,title='With Dropout')
plt.subplot(1,3,3)
pca_figure_all(np.log(nn_output+1),pca_n=2,cluster_n=2,title='NISC')
plt.tight_layout()
plt.savefig('pca.pdf', format='pdf')

f=plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
tsne_figure(np.log(data_gt+1),perplexity=30,cluster_n=2,title='Ground Truth')
plt.subplot(1,3,2)
tsne_figure(np.log(data_count+1),perplexity=30,cluster_n=2,title='With Dropout')
plt.subplot(1,3,3)
tsne_figure(np.log(nn_output+1),perplexity=30,cluster_n=2,title='NISC')
plt.tight_layout()
plt.savefig('tsne.pdf', format='pdf')
plt.show()

f=plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
umap_figure(np.log(data_gt+1),perplexity=30,cluster_n=2,title='Ground Truth')
plt.subplot(1,3,2)
umap_figure(np.log(data_count+1),perplexity=30,cluster_n=2,title='With Dropout')
plt.subplot(1,3,3)
umap_figure(np.log(nn_output+1),perplexity=30,cluster_n=2,title='NISC')
plt.tight_layout()
plt.savefig('umap.pdf', format='pdf')
plt.show()
