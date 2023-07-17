import numpy as np
import scanpy as sc
import anndata
import scvi
from scib_metrics.benchmark import Benchmarker
import torch
from functions import *
import scib_metrics
from lightning.pytorch.callbacks import ModelCheckpoint

adata = anndata.read_h5ad(filename="data/adataLung4.h5ad")
scvi.model.SCVI.setup_anndata(adata, layer="counts")
vaeSD = scvi.model.SCVI(adata)
checkpoint_callback = ModelCheckpoint(dirpath= "models/test/",save_last=True,filename="{epoch:02d}",every_n_epochs=10)
vaeSD.train(default_root_dir="models/test/",enable_checkpointing = True,  max_epochs = 20, callbacks=[checkpoint_callback])