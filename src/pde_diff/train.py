import os
import hydra
import torch

import lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from pde_diff.utils import DatasetRegistry, LossRegistry, unique_id
from pde_diff.model import DiffusionModel
from pde_diff.callbacks import SaveBestModel
import pde_diff.callbacks
from pde_diff.data.utils import split_dataset

@hydra.main(version_base=None, config_name="config.yaml", config_path="../../configs")
def train(cfg: DictConfig):
    hp_config = cfg.experiment.hyperparameters
    if cfg.get("k_folds") is not None:
        assert cfg.id is not None, "If k_folds is used, an id must be provided."

    cfg.id = unique_id(length=5) if cfg.id == None else cfg.id
    cfg.id += f"-{cfg.idx_fold}" if cfg.get("k_folds", None) else ""
    pl.seed_everything(cfg.seed)
    cfg.model.dims = cfg.dataset.dims

    dataset = DatasetRegistry.create(cfg.dataset)
    model = DiffusionModel(cfg)

    if ckpt_path:=cfg.model.get("ckpt_path", None):
        # load weight parameters
        if cfg.get("k_folds", None):
            cfg.id = "-".join(cfg.id.split("-")[:-1])+ "-retrain" +  f"-{cfg.idx_fold}"
        else:
            cfg.id += "-retrain"
        
        map_loc = "cuda" if torch.cuda.is_available() else "cpu"

        if cfg.get("k_folds", None):
            ckpt_path+= f"-{cfg.idx_fold}" +"/best-val_loss.ckpt"
        else:
            ckpt_path+="/best-val_loss.ckpt"
        print(f"Loading model from checkpoint {ckpt_path}")
        model = DiffusionModel.load_from_checkpoint(ckpt_path, cfg=cfg)

    if cfg.get("val_dataset", None):
        dataset_train = dataset
        dataset_val = DatasetRegistry.create(cfg.val_dataset)
    else:
        dataset_train, dataset_val = split_dataset(cfg, dataset)

    # To not get out-of-memory error, accumulate the gradients for batch sizes above 32
    batch_size = hp_config.batch_size
    accumulate_no_batches = 1
    if batch_size>32:
        accumulate_no_batches=hp_config.batch_size//32
        batch_size = 32

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    if dataset_val:
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    
    wandb_name = f"{cfg.experiment.name}-{cfg.id}"

    acc = "gpu" if torch.cuda.is_available() else "cpu"

    if cfg.wandb:
        logger = pl.pytorch.loggers.WandbLogger(name=wandb_name, entity="franka-ppo", project="pde-diff", config=OmegaConf.to_container(cfg.experiment, resolve=True))
    else:
        os.makedirs("logs", exist_ok=True)
        logger = pl.pytorch.loggers.CSVLogger("logs", name=wandb_name)
    
    trainer = pl.Trainer(
        accelerator=acc,
        max_epochs=hp_config.max_epochs,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=hp_config.log_every_n_steps,
        callbacks=[SaveBestModel()],
        accumulate_grad_batches=accumulate_no_batches
    )

    print(f"Starting training of model {cfg.id}")
    if dataset_val:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.fit(model, train_dataloader)
    print(f"Training completed of model {cfg.id}")

if __name__ == "__main__":
    train()
