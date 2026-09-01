import torch.nn as nn
import torch
from lightning.pytorch.callbacks import Callback
from pde_diff.utils import CallbackRegistry

class SaveBestModel(Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_epoch_end(self, trainer, pl_module):
        # If no val loss save model every time
        metric = pl_module.trainer.callback_metrics.get("val_loss")
        if metric is None:
            tag = f"best-val_loss"
            ckpt_path = pl_module.save_dir / f"{tag}.ckpt"
            pl_module.trainer.save_checkpoint(ckpt_path)
            torch.save(pl_module.state_dict(), pl_module.save_dir / f"{tag}-weights.pt")
            print(f"Saved best model weights to {pl_module.save_dir / f'{tag}-weights.pt'}")
            pl_module._save_cfg(pl_module.save_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module.save_model or trainer.global_step <= 0:
            return

        metric = pl_module.trainer.callback_metrics.get("val_loss")
        if metric is None:
            return

        current = float(metric.detach().cpu())

        if pl_module.best_score is None or current < pl_module.best_score:
            pl_module.best_score = current
            tag = f"best-val_loss"
            ckpt_path = pl_module.save_dir / f"{tag}.ckpt"
            pl_module.trainer.save_checkpoint(ckpt_path)
            torch.save(pl_module.state_dict(), pl_module.save_dir / f"{tag}-weights.pt")
            print(f"Saved best model weights to {pl_module.save_dir / f'{tag}-weights.pt'}")
            pl_module._save_cfg(pl_module.save_dir)

            pl_module.log("best_model_improved", 1.0, prog_bar=True)
