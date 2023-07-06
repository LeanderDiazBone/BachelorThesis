import numpy as np
import matplotlib.pyplot as plt
import scvi
import torch

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

def trainModelPostVis(adata,prior,max_epochs,freq=5,save=None, prior_kwargs=None):
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata,prior_distribution=prior, prior_kwargs=prior_kwargs,n_latent=2)
    model.train(max_epochs = max_epochs, check_val_every_n_epoch=freq)# trainer_kwargs=
    if save != None:
        model.save(save)
    return model

def plotPosterior(adata, vae,num=500):
    data = torch.tensor(adata.X[0:num].todense())
    data = data.to(torch.device('cuda:0'))
    distrs, zs = vae.module.z_encoder(data)
    d = np.transpose(np.array(zs.detach().cpu()))
    plt.scatter(d[0],d[1])

def contourPlotDist(dist, xlim, ylim):
    x = np.linspace(-xlim, xlim, 100)
    y = np.linspace(-ylim, ylim, 100)
    X, Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            Z[i][j] = dist.log_prob(torch.tensor([x[i],y[j]]).to(torch.device('cuda:0')))
    plt.contour(X,Y,Z)

def posteriorVisualization(adata, vae, dist, lim, pr):
    plotPosterior(adata, vae)
    contourPlotDist(dist, lim, lim)
    plt.title("Posterior and Prior Vis " + pr + " Prior")
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
    plotPosterior(adata, vae, num = 1000)
    plt.title("Posterior and Prior Samples "+prior+" Prior")

### Functions for Benchmark
from scib_metrics.benchmark import Benchmarker

def trainModelBenchmark(adata, prior, prior_kwargs = None, max_epochs = 100, save=None):
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
    vae = scvi.model.SCVI(adata, prior_distribution = prior,prior_kwargs=prior_kwargs, n_layers=2, n_latent=30)
    vae.train(max_epochs=max_epochs,check_val_every_n_epoch=5)
    if save != None:
        vae.save(save)
    adata.obsm["scVI"] = vae.get_latent_representation()
    return adata, vae

def plotBenchmarkResults(adata,keys=None):
    if keys == None:
        keys = ["Unintegrated", "LIGER", "Scanorama", "scVI"]
    bm = Benchmarker(
    adata,
    batch_key="batch",
    label_key="cell_type",
    embedding_obsm_keys=keys,
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