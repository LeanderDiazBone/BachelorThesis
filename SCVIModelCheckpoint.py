from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Any, Dict, Optional, Set
from lightning.fabric.utilities.types import _PATH
from datetime import timedelta
import scvi
from weakref import proxy
import os
import anndata
import torch

class SCVIModelCheckpoint(ModelCheckpoint):

    def __init__(self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        scviModel: scvi.model.SCVI = None,
        adata: anndata = None
        ):
        super().__init__(
        dirpath,
        filename,
        monitor,
        verbose,
        save_last,
        save_top_k,
        save_weights_only,
        mode,
        auto_insert_metric_name,
        every_n_train_steps,
        train_time_interval,
        every_n_epochs,
        save_on_train_epoch_end)
        self.scviModel = scviModel
        self.adata = adata
    """def save(
        self,
        dir_path: str,
        prefix: Optional[str] = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide another directory for saving.".format(
                    dir_path
                )
            )

        file_name_prefix = prefix or ""
        if save_anndata:
            file_suffix = ""
            if isinstance(self.adata, AnnData):
                file_suffix = "adata.h5ad"
            elif isinstance(self.adata, MuData):
                file_suffix = "mdata.h5mu"
            self.adata.write(
                os.path.join(dir_path, f"{file_name_prefix}{file_suffix}"),
                **anndata_write_kwargs,
            )

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}model.pt")

        # save the model state dict and the trainer state dict only
        model_state_dict = self.module.state_dict()

        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()

        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}

        torch.save(
            {
                "model_state_dict": model_state_dict,
                "var_names": var_names,
                "attr_dict": user_attributes,
            },
            model_save_path,
        )
    """
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        self.scviModel.is_trained = True
        self.scviModel.save(filepath[:-5]+"/")
        self.scviModel.is_trained = False
        trainer.save_checkpoint(filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))