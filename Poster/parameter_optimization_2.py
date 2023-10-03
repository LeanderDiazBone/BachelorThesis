import numpy as np
import scanpy as sc
import anndata
import scvi
from scib_metrics.benchmark import Benchmarker
import torch
import scib_metrics
from lightning.pytorch.callbacks import ModelCheckpoint
import copy
#%matplotlib inline
import sys
sys.path.append('..')
from functions import *
from SCVIModelCheckpoint import SCVIModelCheckpoint 

adata = anndata.read_h5ad(filename="../data/adataImmHum4.h5ad")
ks = [2, 5, 10, 50, 100, 200, 500]
print(adata)

train_adata = adata[:int(33506*0.9)].copy()
test_adata = adata[int(33506*0.9):].copy()
scvi.model.SCVI.setup_anndata(test_adata, layer="counts",batch_key="batch")

def hyperTraining2(adata, prior, hyperpar, ks, n_epochs_kl_warmup=300, max_epochs=20, beta = 5, early_stopping=False):
    adatac = adata.copy()
    keys = []
    vaes = []
    for k in ks:
        adataPrior, vae = trainModel(adata, prior, beta, n_epochs_kl_warmup=n_epochs_kl_warmup,max_epochs=max_epochs,early_stopping=early_stopping,save=f"models/Hyper_Exp/HyperExp_{prior}_{k}",prior_kwargs={hyperpar:k},log=True,logname=f"hyperTrainingMGk={k}")
        keys.append(f"scVI_k={k}")
        adatac.obsm[f"scVI_k={k}"] = vae.get_latent_representation()
        vaes.append(vae)
    return adatac, keys, vaes

adataMG, keysMG, vaes = hyperTraining2(train_adata, "mixofgaus", "k", ks)
bmMG = runBenchmark(adataMG, keysMG, nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans = True, silhouette_label=True, silhouette_batch = True, kbet_per_label=True, label_key='final_annotation')
results = bmMG.get_results(min_max_scale=False)
results.to_csv("results/parameter_optimization_MG_2.csv")
#vaes = []
#results = pd.read_csv("results/parameter_optimization_MG_2.csv")
lls = []
for i in range(len(ks)):
    lls.append(vaes[i].get_marginal_ll(test_adata))
results.drop('Metric Type', axis=1)
results["loglik"] = lls
results["k"] = ks

plot = sns.relplot(data = results, x="loglik", y="Total")
for i in range(results.shape[0]):
    plt.annotate(results["k"][i],(results["loglik"][i],results["Total"][i]+0.001))
plot.set(title="Parameter Tuning")
plot.savefig("plots/prior_optimization_2")