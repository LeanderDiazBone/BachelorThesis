import numpy as np
import matplotlib.pyplot as plt
import scvi
import scanpy as sc
import torch
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import scanorama
import pyliger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import seaborn as sns
import pandas as pd
import copy
### Functions for umap visualization

def trainModelVisualization(adata,prior,max_epochs,freq=5,save=None, prior_kwargs=None, max_kl_weight = 1, n_epochs_kl_warmpup = 100):
    model = scvi.model.SCVI(adata,prior_distribution=prior, prior_kwargs=prior_kwargs)
    model.train(max_epochs=max_epochs,check_val_every_n_epoch=freq,plan_kwargs={"max_kl_weight":max_kl_weight,"n_epochs_kl_warmup":n_epochs_kl_warmpup})
    if save != None:
        model.save(save)
    return model


def umapVisualization(model, adata):
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    adata.obsm["X_scVI"] = model.get_latent_representation()
    #adata_subset = adata[adata.obs.cell_type == "Fibroblast"]
    #latent_subset = model.get_latent_representation(adata_subset)
    #denoised = model.get_normalized_expression(adata_subset, library_size=1e4)
    #denoised.iloc[:5, :5]
    #adata.layers["scvi_normalized"] = model.get_normalized_expression(library_size=10e4)
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(
        adata,
        color=["cell_type"],
        frameon=False,
    )
    """sc.pl.umap(
        adata,
        color=["donor", "cell_source"],
        ncols=2,
        frameon=False,
    )"""

### Functions for visualizing Training History

def plotTrainingHistory(model):
    train_elbo = model.history["elbo_train"][1:]
    test_elbo = model.history["elbo_validation"]
    ax = train_elbo.plot()
    test_elbo.plot(ax=ax)

def plotReconstructionLoss(model):
    train_elbo = model.history["reconstruction_loss_train"][1:]
    test_elbo = model.history["reconstruction_loss_validation"]
    ax = train_elbo.plot()
    test_elbo.plot(ax=ax)

def plotKLLocalLoss(model):
    train_elbo = model.history["kl_local_train"][1:]
    test_elbo = model.history["kl_local_validation"]
    ax = train_elbo.plot()
    test_elbo.plot(ax=ax)

def plotKLGlobalLoss(model):
    train_elbo = model.history["kl_global_train"][1:]
    test_elbo = model.history["kl_global_validation"]
    ax = train_elbo.plot()
    test_elbo.plot(ax=ax)

def plotAllHistory(model):
    fig, ax = plt.subplots(1,3,figsize=(10, 3),layout="constrained")
    train_elbo = model.history["elbo_train"][1:]
    test_elbo = model.history["elbo_validation"]
    ax[0].plot(train_elbo,label="Train")
    ax[0].plot(test_elbo,label="Test")
    ax[0].legend()
    ax[0].set_title("Elbo")
    train_elbo = model.history["reconstruction_loss_train"][1:]
    test_elbo = model.history["reconstruction_loss_validation"]
    ax[1].plot(train_elbo,label="Train")
    ax[1].plot(test_elbo,label="Test")
    ax[1].legend()
    ax[1].set_title("Reconstruction")
    train_elbo = model.history["kl_local_train"][1:]
    test_elbo = model.history["kl_local_validation"]
    ax[2].plot(train_elbo,label="Train")
    ax[2].plot(test_elbo,label="Test")
    ax[2].legend()
    ax[2].set_title("KL local")
    plt.show()

### Functions for Posterior Visualization

def trainModelPostVis(adata,prior,max_epochs=400,n_epochs_kl_warmup=300,freq=5,beta=5,save=None, prior_kwargs=None, log=False,early_stopping=False,logname=""):
    logger = None
    if log:
        logger = TensorBoardLogger(save_dir="lightning_logs",name=logname)
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata,prior_distribution=prior, prior_kwargs=prior_kwargs,n_latent=2)
    model.train(max_epochs = max_epochs, check_val_every_n_epoch=freq,logger=logger,early_stopping=early_stopping, plan_kwargs={"max_kl_weight":beta,"n_epochs_kl_warmup":n_epochs_kl_warmup})
    if save != None:
        model.save(save)
    return model

def getcontourPlotDistPoints(dist, xlim, ylim,flow=False):
    x = np.linspace(-xlim, xlim, 100)
    y = np.linspace(-ylim, ylim, 100)
    X, Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if flow:
                Z[i][j] = dist.log_prob(torch.tensor([[float(x[i]),float(y[j])]]).to(torch.device('cuda:0')))
            else:
                Z[i][j] = dist.log_prob(torch.tensor([x[i],y[j]]).to(torch.device('cuda:0')))
    levels = np.linspace(-15.0, 0.0, 15)
    np.insert(levels, 0,-1000)
    return X, Y, Z, levels

def contourPlotDist(dist, xlim, ylim,flow=False):
    X, Y, Z, levels = getcontourPlotDistPoints(dist, xlim, ylim, flow)
    plt.contourf(X,Y,Z,levels=levels)
    

def getPosteriorPoints(adata, vae, num = 500,cuda=True):
    rc = np.random.choice(adata.X.shape[0], num, replace=False)
    data = torch.tensor(adata.X[rc].todense())
    cell_types = adata.obs["cell_type"][rc]
    if cuda:
        data = data.to(torch.device('cuda:0'))
    distrs, zs = vae.module.z_encoder(data)
    d = np.transpose(np.array(zs.detach().cpu()))
    return d, cell_types


def plotPosterior(d, cell_types, plot_cell_color = True):
    df = pd.DataFrame(np.transpose(np.vstack((d,cell_types))), columns=["dx","dy","cell_type"])
    if plot_cell_color:
        for ct in df["cell_type"].unique():
            plt.scatter(df[df["cell_type"]==ct]["dx"], df[df["cell_type"]==ct]["dy"], s= 5, label=ct)
    else:
        plt.scatter(df["dx"],df["dy"],color="black",s=5)
    plt.plot()

def posteriorVisualization(adata, vae, pr,flow=False):
    d, cell_type = getPosteriorPoints(adata, vae)
    lim = max(d.min(), d.max(), key=abs)
    contourPlotDist(vae.module.prior, lim, lim,flow)
    plotPosterior(d, cell_type)
    plt.title(pr + " Prior")
    plt.show()

def getSamplesDist(distr, num, numsamples = True, flow = False):
    x = []; y = []
    for i in range(num):
        if flow:
            s1, _  = distr.sample(1)
            s = s1.cpu().detach()
            x.append(s[0,0].item())
            y.append(s[0,1].item())
        else:
            if numsamples:
                s = distr.sample(1).cpu()
                x.append(s[0,0])
                y.append(s[0,1])
            else:
                s = distr.sample()
                x.append(s[0])
                y.append(s[1])
    return x,y

def plotSamples(distr, num, title, numsamples = True):
    x,y = getSamplesDist(distr, num, numsamples)
    plt.scatter(x,y)
    plt.title(title)    

def bothVisualizations(adata, vae, prior, lim=5):
    posteriorVisualization(adata, vae, vae.module.prior, lim, prior)
    plotSamples(vae.module.prior,num = 1000,title=prior)
    plotPosterior(getPosteriorPoints(adata, vae, num = 1000))
    plt.title("Posterior and Prior Samples "+prior+" Prior")

def plotFlowSamples(vaeNF, num = 500):
    x, y = getSamplesDist(vaeNF.module.prior, num, flow =True)
    fig, ax = plt.subplots()
    scatter = ax.scatter(x,y)
    plt.show()

def posteriorVisualizationAll(adata, vaeSN, vaeMG, vaeVP, vaeNF, sample = False):
    dSN, cell_typesSN = getPosteriorPoints(adata, vaeSN, num=1000)
    limSN = max(dSN.min(), dSN.max(), key=abs)
    XSN, YSN, ZSN, levelsSN = getcontourPlotDistPoints(vaeSN.module.prior, limSN, limSN,flow=False)
    xSN, ySN = getSamplesDist(vaeSN.module.prior, 1000)
    dMG, cell_typesMG = getPosteriorPoints(adata, vaeMG, num=1000)
    limMG = max(dMG.min(), dMG.max(), key=abs)
    XMG, YMG, ZMG, levelsMG = getcontourPlotDistPoints(vaeMG.module.prior, limMG, limMG,flow=False)
    xMG, yMG = getSamplesDist(vaeMG.module.prior, 1000)
    dVP, cell_typesVP = getPosteriorPoints(adata, vaeVP, num=1000)
    limVP = max(dVP.min(), dVP.max(), key=abs)
    XVP, YVP, ZVP, levelsVP = getcontourPlotDistPoints(vaeVP.module.prior, limVP, limVP,flow=False)
    xVP, yVP = getSamplesDist(vaeVP.module.prior, 1000)
    dNF, cell_typesNF = getPosteriorPoints(adata, vaeNF, num=1000)
    limNF = max(dNF.min(), dNF.max(), key=abs)
    XNF, YNF, ZNF, levelsNF = getcontourPlotDistPoints(vaeNF.module.prior, limNF, limNF,flow=True)
    xNF, yNF = getSamplesDist(vaeNF.module.prior, 1000, flow=True)
    fig, axs = plt.subplots(2,2,figsize=(16,16))
    
    if sample:
        axs[0,0].scatter(xSN,ySN)
    else:
        axs[0,0].contourf(XSN,YSN,ZSN,levels=levelsSN)
    dfSN = pd.DataFrame(np.transpose(np.vstack((dSN,cell_typesSN))), columns=["dx","dy","cell_type"])
    for ct in dfSN["cell_type"].unique():
        axs[0,0].scatter(dfSN[dfSN["cell_type"]==ct]["dx"], dfSN[dfSN["cell_type"]==ct]["dy"], s= 5, label=ct)
    axs[0,0].set_title("Standard Normal Prior")
    
    if sample:
        axs[0,1].scatter(xMG,yMG)
    else:
        axs[0,1].contourf(XMG,YMG,ZMG,levels=levelsMG)
    dfMG = pd.DataFrame(np.transpose(np.vstack((dMG,cell_typesMG))), columns=["dx","dy","cell_type"])
    for ct in dfSN["cell_type"].unique():
        axs[0,1].scatter(dfMG[dfMG["cell_type"]==ct]["dx"], dfMG[dfMG["cell_type"]==ct]["dy"], s= 5, label=ct)
    axs[0,1].set_title("Mixture of Gaussians Prior")
    
    if sample:
        axs[1,0].scatter(xVP,yVP)
    else:
        axs[1,0].contourf(XVP,YVP,ZVP,levels=levelsVP)
    dfVP = pd.DataFrame(np.transpose(np.vstack((dVP,cell_typesVP))), columns=["dx","dy","cell_type"])
    for ct in dfSN["cell_type"].unique():
        axs[1,0].scatter(dfVP[dfVP["cell_type"]==ct]["dx"], dfVP[dfVP["cell_type"]==ct]["dy"], s= 5, label=ct)
    axs[1,0].set_title("Vamp Prior")
    
    if sample:
        axs[1,1].scatter(xNF,yNF)
    else:
        axs[1,1].contourf(XNF,YNF,ZNF,levels=levelsNF)
    dfNF = pd.DataFrame(np.transpose(np.vstack((dNF,cell_typesNF))), columns=["dx","dy","cell_type"])
    for ct in dfSN["cell_type"].unique():
        axs[1,1].scatter(dfNF[dfNF["cell_type"]==ct]["dx"], dfNF[dfNF["cell_type"]==ct]["dy"], s= 5, label=ct)
    axs[1,1].set_title("Normal Flow Prior")
    #axs.set_xlabel("latent_1")
    #axs.set_ylabel("latent_2")
    plt.show()

### Functions for Benchmark
from scib_metrics.benchmark import Benchmarker
import scib_metrics
from lightning.pytorch.callbacks import Timer

def trainModelBenchmark(adata, prior,  max_epochs = 400,n_epochs_kl_warmup=300,beta=5, save=None, batch_key="batch",log=False,logname="",early_stopping=False):
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    logger = None
    if log:
        logger = TensorBoardLogger(save_dir="lightning_logs",name=logname)
    vae = scvi.model.SCVI(adata, prior_distribution = prior, n_latent=10)
    vae.train(max_epochs=max_epochs,check_val_every_n_epoch=5,logger=logger,early_stopping=early_stopping,plan_kwargs={"max_kl_weight":beta,"n_epochs_kl_warmup":n_epochs_kl_warmup})
    adata.obsm["scVI"] = vae.get_latent_representation()
    if save != None:
        vae.save(save, save_anndata=True)
    return adata, vae

def plotBenchmarkResults(adata,keys=None,label_key="cell_type",batch_key="batch",save_dir=None):
    if keys == None:
        keys = ["Unintegrated", "LIGER", "Scanorama", "scVI"]
    bm = runBenchmark(adata, keys, label_key=label_key, batch_key=batch_key,isolated_labels = True, nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans = True, silhouette_label=True, clisi_knn = True, graph_connectivity=True, ilisi_knn=True, kbet_per_label=True, pcr_comparison=True, silhouette_batch=True)
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False,save_dir=save_dir)
    plt.show()


def scanoramaPredict(adata,batch_label="batch"):
    batch_cats = adata.obs[batch_label].cat.categories
    adata_list = [adata[adata.obs[batch_label] == b].copy() for b in batch_cats]
    scanorama.integrate_scanpy(adata_list)

    adata.obsm["Scanorama"] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata.obsm["Scanorama"][adata.obs[batch_label] == b] = adata_list[i].obsm["X_scanorama"]


def ligerPredict(adata,batch_label="batch"):
    batch_cats = adata.obs[batch_label].cat.categories
    bdata = adata.copy()
    # Pyliger normalizes by library size with a size factor of 1
    # So here we give it the count data
    bdata.X = bdata.layers["counts"]
    # List of adata per batch
    adata_list = [bdata[bdata.obs.batch == b].copy() for b in batch_cats]
    for i, ad in enumerate(adata_list):
        ad.uns["sample_name"] = batch_cats[i]
        # Hack to make sure each method uses the same genes
        ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)


    liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
    # Hack to make sure each method uses the same genes
    liger_data.var_genes = bdata.var_names
    pyliger.normalize(liger_data)
    pyliger.scale_not_center(liger_data)
    pyliger.optimize_ALS(liger_data, k=30)
    pyliger.quantile_norm(liger_data)


    adata.obsm["LIGER"] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata.obsm["LIGER"][adata.obs.batch == b] = liger_data.adata_list[i].obsm["H_norm"]


### latent space

def plotW(vae):
    plt.bar(np.linspace(1,20,20),torch.softmax(vae.module.prior.w,dim=0).detach().cpu())

def covariance(adata, vae):
    dist, z = vae.module.z_encoder(torch.tensor(adata.X.todense()).to("cuda:0"))
    return torch.cov(z.T)

def activateUnits(adata,vae):
    arr = np.diag(np.array(covariance(adata, vae).cpu().detach()))
    return arr, (arr > 0.1).sum()


def plotcovarianceMatrix(adata, vae):
    ax = sns.heatmap(covariance(adata,vae).detach().cpu(),vmax=1,vmin=-1,cmap=sns.color_palette("vlag", as_cmap=True))
    ax.set(yticklabels=[])
    ax.tick_params(left=False)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    plt.show()


### Metrics Epochs experiment

import os
def loadCkptToSCVIModel(vae, adata, dir,max_epochs,n_eps = 10):
    for i in range(int(max_epochs/n_eps)):
        state_dict = torch.load(f"{dir}epoch={n_eps*i+n_eps-1}.ckpt")["state_dict"]
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key[7:]] = state_dict.pop(key)
            
        var_names = adata.var_names.astype(str)
        var_names = var_names.to_numpy()

        # get all the user attributes
        user_attributes = vae._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        model_save_path = os.path.join(f"{dir}epoch={n_eps*i+n_eps-1}", f"model.pt")
        os.mkdir(f"{dir}epoch={n_eps*i+n_eps-1}")
        torch.save(
                    {
                        "model_state_dict": state_dict,
                        "var_names": var_names,
                        "attr_dict": user_attributes,
                    },
                    model_save_path
        )

### KL-Metrics Experiment

def betasTraining(vae, adata, betas, n_epochs_kl_warmup, max_epochs, early_stopping=False):
    adatac = adata.copy()
    keys = []
    for beta in betas:
        vae_cur = copy.deepcopy(vae)
        vae_cur.train(plan_kwargs={"max_kl_weight":beta,"n_epochs_kl_warmup":n_epochs_kl_warmup},max_epochs=max_epochs,early_stopping=early_stopping)
        keys.append(f"scVI_beta={beta}")
        adatac.obsm[f"scVI_beta={beta}"] = vae_cur.get_latent_representation()
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