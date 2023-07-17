import numpy as np
import scanpy as sc
import scvi
from functions import *
from lightning.pytorch.callbacks import ModelCheckpoint
import anndata

adata = anndata.read_h5ad(filename="/local/data/BachelorThesis/data/adataLung4.h5ad")
scvi.model.SCVI.setup_anndata(adata, layer="counts")
vaeSD = scvi.model.SCVI(adata)
checkpoint_callback = ModelCheckpoint(dirpath= "/local/data/BachelorThesis/models/test/",save_last=False,filename="{epoch:d}",every_n_epochs=10,save_top_k=-1)
vaeSD.train(enable_checkpointing = True,  max_epochs = 100, callbacks=[checkpoint_callback], check_val_every_n_epoch=1)