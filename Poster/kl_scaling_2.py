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
import seaborn as sns

"""
def betasTraining(adata, prior, betas, n_epochs_kl_warmup=300, max_epochs=400, early_stopping=False):
    keys = []
    adatac = adata.copy()
    for beta in betas:
        adataPrior, vae  = trainModel(adata, prior, beta, n_epochs_kl_warmup=n_epochs_kl_warmup,max_epochs=max_epochs,early_stopping=early_stopping,save=f"models/KL_Exp/{prior}_{beta}")
        keys.append(f"scVI_beta={beta}")
        adatac.obsm[f"scVI_beta={beta}"] = vae.get_latent_representation()
    return adatac, keys

def runBenchmark(adata, keys, batch_key="batch", label_key="cell_type", isolated_labels = False, nmi_ari_cluster_labels_leiden=False, nmi_ari_cluster_labels_kmeans = False, silhouette_label=False, clisi_knn = False, graph_connectivity=False, ilisi_knn=False, kbet_per_label=False, pcr_comparison=False, silhouette_batch=False):
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=keys,
        batch_correction_metrics=scib_metrics.benchmark.BatchCorrection(silhouette_batch, ilisi_knn, kbet_per_label, graph_connectivity, pcr_comparison),
        bio_conservation_metrics=scib_metrics.benchmark.BioConservation(isolated_labels, nmi_ari_cluster_labels_leiden, nmi_ari_cluster_labels_kmeans, silhouette_label, clisi_knn),
        n_jobs=16,
    )
    bm.benchmark()
    return bm
"""
adata = anndata.read_h5ad(filename="../data/adataImmHum4.h5ad")
print(adata)

betas = [0, 1, 2, 4, 6, 8, 10, 15]
adataMG, keysMG = betasTraining(adata,"mixofgaus", betas)
adatac = adata.copy()
"""for i in range(8):
    vae = scvi.model.SCVI.load(f"models/KL_Exp_0/mixofgaus_{betas[i]}",adata=adata)
    keys.append(f"beta={betas[i]}")
    adatac.obsm[keys[i]] = vae.get_latent_representation()
"""
bmMG = runBenchmark(adataMG, keysMG, nmi_ari_cluster_labels_leiden=True, kbet_per_label=True,label_key='final_annotation')
results = bmMG.get_results(min_max_scale=False)
results.to_csv("results/kl_scaling_MG_adataHumPan.csv")
results = results.drop("Metric Type")
results["beta"] = betas
plot = sns.relplot(data = results, x="Batch correction", y="Bio conservation")
for i in range(results.shape[0]):
    plt.annotate(results["beta"][i],(results["Batch correction"][i],results["Bio conservation"][i]+0.005))
plot.set(title="KL-term Scaling")
plot.savefig("plots/kl_scaling_2")
