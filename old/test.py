import numpy as np
import scanpy as sc
import scvi
from functions import *
from lightning.pytorch.callbacks import ModelCheckpoint
import anndata
from SCVIModelCheckpoint import SCVIModelCheckpoint 
""""
adata = anndata.read_h5ad(filename="/local/data/BachelorThesis/data/adataLung4.h5ad")
scvi.model.SCVI.setup_anndata(adata, layer="counts")
vaeSD = scvi.model.SCVI(adata)
checkpoint_callback = ModelCheckpoint(dirpath= "/local/data/BachelorThesis/models/test/",save_last=False,filename="{epoch:d}",every_n_epochs=10,save_top_k=-1)
vaeSD.train(enable_checkpointing = True,  max_epochs = 100, callbacks=[checkpoint_callback], check_val_every_n_epoch=1)
"""

adata = anndata.read_h5ad(filename="/local/data/BachelorThesis/data/adataLung4.h5ad")
scvi.model.SCVI.setup_anndata(adata, layer="counts")
def trainMetricsEpochs(vae, folder, max_epochs, every_n = 10, log=False,logname="", early_stopping=False):
    logger = None
    if log:
        logger = TensorBoardLogger(save_dir="lightning_logs",name=logname)
    checkpoint_callback = SCVIModelCheckpoint(dirpath= f"/local/data/BachelorThesis/models/{folder}/",save_last=False,filename="{epoch:d}",every_n_epochs=every_n,save_top_k=-1,scviModel=vae)
    vae.train(enable_checkpointing = True,  max_epochs = max_epochs, callbacks=[checkpoint_callback], check_val_every_n_epoch=1,logger=logger,early_stopping=early_stopping)

vaeSD = scvi.model.SCVI(adata,prior_distribution='sdnormal')
trainMetricsEpochs(vaeSD, "MetEp_SD", 5, every_n=1)
vae = scvi.model.SCVI.load(f"models/MetEp_SD/epoch={3}/",adata=adata)
print(vae)