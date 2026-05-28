import torch
from torch.utils.data._utils.collate import default_collate
import einops as ein
import numpy as np
import lightning as pl
from diffusers import UNet2DModel, UNet2DConditionModel
from omegaconf import OmegaConf
from pathlib import Path

from pde_diff.data.datasets import increment_clock_features

from pde_diff.utils import SchedulerRegistry, LossRegistry, ModelRegistry, init_means_and_stds_era5
import pde_diff.scheduler
import pde_diff.loss
from pde_diff.data import datasets
from pde_diff.unet_model import Unet3D


class DiffusionModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.scheduler = SchedulerRegistry.create(cfg.scheduler)
        self.loss_fn = LossRegistry.create(cfg.loss)
        self.hp_config = cfg.experiment.hyperparameters
        self.model = ModelRegistry.create([cfg.model, self.hp_config])
        if cfg.dataset.name == 'era5' and cfg.loss.name == 'vorticity': #semi cursed (TODO clean up)
            self.atmospheric_features = cfg.dataset.atmospheric_features
            self.single_features = cfg.dataset.single_features
            self.static_features = cfg.dataset.static_features
            self.means, self.stds, self.diff_means, self.diff_stds = init_means_and_stds_era5(self.atmospheric_features,
                                                                                              self.single_features,
                                                                                              self.static_features)
            self.loss_fn.set_mean_and_std(self.means, self.stds, self.diff_means, self.diff_stds)

        self.data_dims = cfg.dataset.dims

        self.conditional = cfg.dataset.time_series # Add conditional flag
        self.cfg = cfg
        self.save_model = cfg.model.save_best_model
        self.mse = torch.nn.MSELoss(reduction='none')
        self.best_score = None
        self.save_dir = Path(cfg.save_dir) / Path(cfg.experiment.name + "-" + cfg.id) if cfg.get("id", None) else None

    def training_step(self, batch, batch_idx):
        if self.conditional:
            # Apply conditional logic here
            conditionals, state = batch
        else:
            state = batch

        noise = torch.randn_like(state)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (state.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(state, noise, steps)

        if self.conditional:
            model_in = torch.cat((conditionals, noisy_images), dim=1)
        else:
            model_in = noisy_images

        model_out = self.model(model_in, steps)

        if self.conditional:
            num_vars = (conditionals.shape[1] - 8) // 6
            x0_hat = torch.cat([conditionals[:, num_vars*3+4:-4], model_out], dim=1) #hardcoded for now (TODO)
        else:
            x0_hat = model_out

        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps] #https://arxiv.org/pdf/2303.09556.pdf
        loss = self.loss_fn(model_out=model_out, target=state, x0_hat=x0_hat, var=variance)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.additional_validation_metrics(model_out, state, x0_hat, steps, validation=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.conditional:
            conditionals, target = batch
        else:
            target = batch
        noise = torch.randn_like(target)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (target.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(target, noise, steps)

        if self.conditional:
            noisy_images = torch.cat([conditionals, noisy_images], dim=1)

        model_out = self.model(noisy_images, steps)
        x0_hat = model_out
        if self.conditional:
            num_vars = (conditionals.shape[1] - 8) // 6
            x0_hat = torch.cat([conditionals[:, num_vars*3+4:-4], model_out], dim=1) #hardcoded for now (TODO)

        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps]
        loss = self.loss_fn(model_out=model_out, target=target, x0_hat=x0_hat, var=variance)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.additional_validation_metrics(model_out, target, x0_hat, steps)

    def test_step(self, batch, batch_idx):
        if self.conditional:
            conditionals, target = batch
        else:
            target = batch

        noise = torch.randn_like(target)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (target.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(target, noise, steps)

        if self.conditional:
            noisy_images = torch.cat([conditionals, noisy_images], dim=1)

        model_out = self.model(noisy_images, steps)
        x0_hat = model_out
        if self.conditional:
            num_vars = (conditionals.shape[1] - 8) // 6
            x0_hat = torch.cat([conditionals[:, num_vars*3+4:-4], model_out], dim=1) #hardcoded for now (TODO)

        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps]
        loss = self.loss_fn(model_out=model_out, target=target, x0_hat=x0_hat, var=variance)
        self.log("test_loss", loss, prog_bar=True, batch_size=model_out.size(0))
        self.additional_validation_metrics(model_out, target,x0_hat, steps)

    def additional_validation_metrics(self, model_out, target, x0_hat, steps, validation=True):
        if validation:
            metrics = self.cfg.dataset.validation_metrics
            for metric_name in metrics:
                if metric_name == "mse":
                    mse = (self.mse(model_out, target) * self.scheduler.p2_loss_weight[steps][:, None, None, None]).mean()
                    self.log("val_mse_(weighted)", mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))                    

                if metric_name == 'era5_vorticity':
                    assert self.cfg.loss.name == 'vorticity', "era5_vorticity metric can only be used with vorticity loss"
                    num_channels = x0_hat.shape[1]
                    x0_previous = x0_hat[:, :num_channels//2, :, :]
                    x0_change_pred = x0_hat[:, num_channels//2:, :, :]
                    residual_planetary = self.loss_fn.compute_residual_planetary_vorticity(x0_previous, x0_change_pred).abs().mean()
                    residual_geo_wind = self.loss_fn.compute_residual_geostrophic_wind(x0_previous, x0_change_pred).abs().mean()
                    self.log("val_era5_planetary_residual(norm)", residual_planetary, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))
                    self.log("val_era5_geo_wind_residual(norm)", residual_geo_wind, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))

                    model_out_reshaped = ein.rearrange(model_out, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
                    target_reshaped = ein.rearrange(target, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
                    for i in range(target_reshaped.shape[2]):
                        mse_var = (self.mse(model_out_reshaped[:,:,i], target_reshaped[:,:,i]) * self.scheduler.p2_loss_weight[steps][:, None, None, None]).mean()
                        self.log(f"val_mse_var_{i}_(weighted)", mse_var, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))
        else:
            metrics = self.cfg.dataset.test_metrics
            for metric_name in metrics:
                if metric_name == "mse":
                    mse = (self.mse(model_out, target) * self.scheduler.p2_loss_weight[steps][:, None, None, None]).mean()
                    self.log("train_mse_(weighted)", mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))

    def on_validation_epoch_end(self):
        """For on_epoch_end validation metrics"""
        metrics = self.cfg.dataset.validation_metrics
        for metric_name in metrics:
            if metric_name == "darcy":
                with torch.no_grad():
                    x0_preds = self.sample_loop(batch_size=16)
                    darcy_res = self.loss_fn.compute_residual(x0_preds).mean().abs()
                self.log("val_darcy_residual", darcy_res, prog_bar=True, on_epoch=True, sync_dist=True)
            if metric_name == 'era5_vorticity':
                with torch.no_grad():
                    val_conditionals = self._uniform_val_batch(n=16)[0].to(self.device)
                    x0_preds = self.sample_loop(batch_size=16, conditionals=val_conditionals)
                    num_vars = (val_conditionals.shape[1] - 8) // 6
                    x0_prev = val_conditionals[:, num_vars*3+4:-4]
                    residual_planetary = self.loss_fn.compute_residual_planetary_vorticity(x0_prev, x0_preds).abs().mean()
                    residual_geo_wind = self.loss_fn.compute_residual_geostrophic_wind(x0_prev, x0_preds).abs().mean()
                self.log("val_era5_sampled_planetary_residual(norm)", residual_planetary, prog_bar=True, on_epoch=True, sync_dist=True)
                self.log("val_era5_sampled_geo_wind_residual(norm)", residual_geo_wind, prog_bar=True, on_epoch=True, sync_dist=True)

    def _uniform_val_batch(self, n=16):
        vdl = self.trainer.val_dataloaders
        vdl = vdl[0] if isinstance(vdl, (list, tuple)) else vdl

        ds = vdl.dataset
        N = len(ds)

        # 16 evenly spaced indices in [0, N-1]
        idx = torch.linspace(0, N - 1, steps=n).long().tolist()

        # fetch items and collate into a batch like the dataloader would
        items = [ds[i] for i in idx]
        batch = default_collate(items)

        return batch

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hp_config.lr, weight_decay=self.hp_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50) # Reset every 50 epochs (TODO make this configurable)
        return [optimizer], [scheduler]

    def forward(self, samples, t):
        """One reverse diffusion step when the model predicts x0 instead of ε."""

        self.model.eval()
        with torch.no_grad():
            t_batch = torch.full((samples.size(0),), t, device=self.device, dtype=torch.long)
            x0_pred = self.model(samples, t_batch)
            if self.conditional:
                samples = samples[:,-self.data_dims.output_dims:,:,:]
            mean = self.scheduler.posterior_mean_coef1[t_batch][:, None, None, None] * x0_pred
            mean.add_(self.scheduler.posterior_mean_coef2[t_batch][:, None, None, None] * samples)
            z = torch.randn_like(samples) if t > 0 else 0
            samples = mean + self.scheduler.sigmas[t] * z
        return samples

    def sample_loop(self, batch_size=1, conditionals=None):
        samples = torch.randn((batch_size,int(self.data_dims.output_dims), int(self.data_dims.x), int(self.data_dims.y)), device=self.device)
        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            if self.conditional:
                model_in = torch.cat((conditionals, samples), dim=1)
            else:
                model_in = samples

            samples = self.forward(model_in, t)
        return samples

    def forecast(self, initial_condition, steps):
        """Forecast multiple steps ahead given an initial condition tensor.
        Returns the predicted changes at each forecast step."""
        self.model.eval()
        current_state = initial_condition.to(self.device)
        forecasted_changes = []
        for step in range(steps):
            with torch.no_grad():
                prediction = self.sample_loop(batch_size=current_state.size(0), conditionals=current_state)
                next_state = current_state[:,-(self.data_dims.input_dims-self.data_dims.output_dims)//2:,:,:].clone().to(self.device)
                next_state[:,:self.data_dims.output_dims, :, :] = self.loss_fn.get_original_states(x0_previous=next_state[:,:self.data_dims.output_dims], x0_change_pred=prediction, rearrange=False)[1]
                next_state[:,:self.data_dims.output_dims] = self.loss_fn.get_normalized_states(x0=next_state[:,:self.data_dims.output_dims, :, :] )
                # Update next_state with time information:
                next_state[:, -4:, :, :] = increment_clock_features(
                    next_state[:, -4:, :, :], step_size=self.cfg.dataset.time_step
                ).to(self.device)
                current_state = torch.cat([current_state[:,-(self.data_dims.input_dims-self.data_dims.output_dims)//2:,:,:], next_state], dim=1)
            forecasted_changes.append(prediction)

        return torch.stack(forecasted_changes, dim=1)

    def ddim_forward(self, samples, t, t_prev): #this assumes model predicts epsilon
        z = torch.randn_like(samples) if t_prev > 0 else 0
        t_batch =torch.tensor([t]*samples.size(0), device=self.device)
        t_prev_batch =torch.tensor([t_prev]*samples.size(0), device=self.device)
        samples = self.scheduler.ddim_sample(self.model(samples, t_batch), t_batch, t_prev_batch, z, samples) + self.scheduler.sigmas[t] * z
        return samples

    def sample_loop_ddim(self, input=None, residual=None, current_time=None, tau_length=100, microbatch=16, use_amp=False):
        device = self.device
        if microbatch is None:
            microbatch = input.size(0)

        ct = current_time.to(torch.long)
        tau = int(tau_length)

        def make_sched(T: int, tau: int):
            s = list(range(0, max(T, 0), tau))
            if not s or s[-1] != T:
                s.append(T)
            return s

        schedules = [make_sched(int(T.item()), tau) for T in ct]
        groups = {}
        for i, s in enumerate(schedules):
            key = tuple(s)
            groups.setdefault(key, []).append(i)

        x = input
        amp_cm = torch.autocast(device_type=device.type if hasattr(device, "type") else "cuda", dtype=torch.float16) if use_amp else torch.cuda.amp.autocast(enabled=False)

        with amp_cm:
            for sched, idxs in groups.items():
                idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
                timesteps = list(sched)

                for t_prev, t in zip(reversed(timesteps[1:]), reversed(timesteps[:-1])):
                    for start in range(0, idx_tensor.numel(), microbatch):
                        sel = idx_tensor[start:start + microbatch]
                        x_sel = torch.index_select(x, 0, sel)
                        x_sel = self.ddim_forward(x_sel, int(t), int(t_prev))
                        x.index_copy_(0, sel, x_sel)
        return x

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def _save_cfg(self, save_dir):
        """Save self.cfg next to checkpoints as YAML (if OmegaConf) or JSON."""
        import json
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(self, "cfg") and (
                getattr(self.cfg, "__class__", None).__name__ in {"DictConfig", "ListConfig"}
                or hasattr(self.cfg, "to_container")
            ):
                (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(self.cfg))
                return
        except Exception:
            pass

@ModelRegistry.register("dummy")
class DummyModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.layer2 = torch.nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x, t):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

@ModelRegistry.register("unet3d")
class UNet3DWrapper(torch.nn.Module):
    def __init__(self, cfg_list):
        super().__init__()
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        self.unet = Unet3D(dim = 32, channels = model_hp.dims.output_dims)

    def forward(self, x, t):
        return self.unet(x, t)

@ModelRegistry.register("unet3d_conditional")
class UNet3DWrapperConditional(torch.nn.Module):
    def __init__(self, cfg_list):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        self.in_channels = int(model_hp.dims.input_dims)
        self.out_channels = int(model_hp.dims.output_dims)
        self.unet = Unet3D(dim = 32, channels=self.out_channels, out_dim=self.out_channels, cond_dim=self.in_channels - self.out_channels).to(device)

    def forward(self, x, t):
        cond_channels = self.in_channels - self.out_channels
        conditionals = x[:, :cond_channels, :, :]
        inputs = x[:, cond_channels:, :, :]
        return self.unet(x = inputs, time = t, cond = conditionals)


@ModelRegistry.register("unet2d")
class UNet2DWrapper(torch.nn.Module):
    def __init__(self, cfg_list):
        super().__init__()
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        in_channels = int(model_hp.dims.input_dims)
        out_channels = int(model_hp.dims.output_dims)
        self.unet = UNet2DModel(
            sample_size=int(model_hp.dims.x),
            in_channels=in_channels,
            out_channels=out_channels,
            dropout = hp_params.dropout,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t):
        return self.unet(sample=x, timestep=t).sample

@ModelRegistry.register("unet2d_conditional")
class UNet2DConditionalWrapper(torch.nn.Module):
    def __init__(self, cfg_list):
        super().__init__()
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        self.in_channels = int(model_hp.dims.input_dims)
        self.out_channels = int(model_hp.dims.output_dims)
        self.unet = UNet2DConditionModel(
            sample_size=(int(model_hp.dims.x), int(model_hp.dims.y)),
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            dropout = hp_params.dropout,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=self.in_channels - self.out_channels,  # Assuming half of input channels are for conditioning
        )

    def forward(self, x, t):
        # Split x into conditionals and actual input
        cond_channels = self.in_channels - self.out_channels
        conditionals = x[:, :cond_channels, :, :]
        batch_size, cond_channels, height, width = conditionals.shape
        conditionals = conditionals.view(batch_size, cond_channels, height * width).permute(0, 2, 1)
        inputs = x[:, cond_channels:, :, :]
        return self.unet(sample=inputs, timestep=t, encoder_hidden_states=conditionals).sample

if __name__ == "__main__":
    pass
