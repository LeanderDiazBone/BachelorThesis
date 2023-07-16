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

### Functions for umap visualization

def trainModelVisualization(adata,prior,max_epochs,freq=5,save=None, prior_kwargs=None):
    model = scvi.model.SCVI(adata,prior_distribution=prior, prior_kwargs=prior_kwargs)
    model.train(max_epochs=max_epochs,check_val_every_n_epoch=freq)
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

def trainModelPostVis(adata,prior,max_epochs,freq=5,save=None, prior_kwargs=None, log=False,early_stopping=False,logname=""):
    logger = None
    if log:
        logger = TensorBoardLogger(save_dir="lightning_logs",name=logname)
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata,prior_distribution=prior, prior_kwargs=prior_kwargs,n_latent=2)
    model.train(max_epochs = max_epochs, check_val_every_n_epoch=freq,logger=logger,early_stopping=early_stopping)
    if save != None:
        model.save(save)
    return model

def contourPlotDist(dist, xlim, ylim,flow=False):
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
    plt.contourf(X,Y,Z,levels=levels)
    
def getPosteriorPoints(adata, vae, num = 500,cuda=True):
    data = torch.tensor(adata.X[np.random.choice(adata.X.shape[0], num, replace=False)].todense())
    if cuda:
        data = data.to(torch.device('cuda:0'))
    distrs, zs = vae.module.z_encoder(data)
    d = np.transpose(np.array(zs.detach().cpu()))
    return d


def plotPosterior(d):
    plt.scatter(d[0],d[1],color="black",s=5)

def posteriorVisualization(adata, vae, pr,flow=False):
    d = getPosteriorPoints(adata, vae)
    lim = max(d.min(), d.max(), key=abs)
    contourPlotDist(vae.module.prior, lim, lim,flow)
    plotPosterior(d)
    plt.title("Posterior and Prior Vis " + pr + " Prior")
    plt.xlabel("latent_1")
    plt.ylabel("latent_2")
    plt.show()

def plotSamples(distr, num, title, numsamples = True):
    x = []; y = []
    for i in range(num):
        if numsamples:
            s = distr.sample(1).cpu()
            x.append(s[0,0])
            y.append(s[0,1])
        else:
            s = distr.sample()
            x.append(s[0])
            y.append(s[1])
    plt.scatter(x,y)
    plt.title(title)    

def bothVisualizations(adata, vae, prior, lim=5):
    posteriorVisualization(adata, vae, vae.module.prior, lim, prior)
    plotSamples(vae.module.prior,num = 1000,title=prior)
    plotPosterior(getPosteriorPoints(adata, vae, num = 1000))
    plt.title("Posterior and Prior Samples "+prior+" Prior")

def plotFlowSamples(vaeNF):
    x = []; y = []
    for i in range(100):
        s1, _  = vaeNF.module.prior.sample(1)
        s = s1.cpu().detach()
        x.append(s[0,0].item())
        y.append(s[0,1].item())
    fig, ax = plt.subplots()
    scatter = ax.scatter(x,y)
    plt.show()

### Functions for Benchmark
from scib_metrics.benchmark import Benchmarker
import scib_metrics

def trainModelBenchmark(adata, prior, prior_kwargs = None, max_epochs = 100, save=None, batch_key="batch",log=False,logname="",early_stopping=False):
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    logger = None
    if log:
        logger = TensorBoardLogger(save_dir="lightning_logs",name=logname)
    vae = scvi.model.SCVI(adata, prior_distribution = prior,prior_kwargs=prior_kwargs, n_layers=2, n_latent=30)
    vae.train(max_epochs=max_epochs,check_val_every_n_epoch=5,logger=logger,early_stopping=early_stopping)
    if save != None:
        sss
    adata.obsm["scVI"] = vae.get_latent_representation()
    return adata, vae

def plotBenchmarkResults(adata,keys=None,label_key="cell_type",batch_key="batch"):
    if keys == None:
        keys = ["Unintegrated", "LIGER", "Scanorama", "scVI"]
    bm = Benchmarker(
    adata,
    batch_key=batch_key,
    label_key=label_key,
    embedding_obsm_keys=keys,
    bio_conservation_metrics=scib_metrics.benchmark.BioConservation(isolated_labels = True, nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans = True, silhouette_label=True, clisi_knn = True),
    n_jobs=6,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False)

def plotBenchmarkResultsAll(sdnormalAdata, normalAdata, mogAdata, vampAdata):
    adataAll = normalAdata
    adataAll.obsm["scVISDNormal"] = sdnormalAdata.obsm["scVI"]
    adataAll.obsm["scVINormal"] = normalAdata.obsm["scVI"]
    adataAll.obsm["scVIMoG"] = mogAdata.obsm["scVI"]
    adataAll.obsm["scVIVamp"] = vampAdata.obsm["scVI"]
    plotBenchmarkResults(adataAll,["Unintegrated", "LIGER", "Scanorama", "scVINormal","scVIMoG","scVIVamp","scVISDNormal"])

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