import time

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info


class PeakVRAMMonitorCallback(Callback):
    """Callback to monitor peak VRAM usage during training."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
            trainer.logger.experiment.log({"peak_vram_gb": peak_memory})
            rank_zero_info(f"Peak VRAM usage: {peak_memory:.2f} GB")


class FLOPSMonitorCallback(Callback):
    """Callback to count FLOPs for the entire MATT method using actual data from datamodule."""

    def __init__(self) -> None:
        self.flops_counted = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.flops_counted:
            try:
                from lightning.fabric.utilities import measure_flops

                sample_batch = next(iter(trainer.datamodule.train_dataloader()))
                batch_size = len(sample_batch["teacher_input_ids"])

                sample_batch = {
                    k: v.to(pl_module.device) if hasattr(v, "to") else v
                    for k, v in sample_batch.items()
                }
                
                model_fwd = lambda: pl_module.common_step(sample_batch, 0)
                model_loss = lambda y: y
                fwd_and_bwd_flops = measure_flops(pl_module, model_fwd, model_loss)

                flops = fwd_and_bwd_flops // batch_size

                trainer.logger.experiment.log({"flops": flops})
                rank_zero_info(f"FLOPS: {flops:,}")

                self.flops_counted = True
                
            except ImportError:
                rank_zero_info("Lightning fabric not available for FLOPs counting")
            except Exception as e:
                rank_zero_info(f"Error counting FLOPs: {e}")
                rank_zero_info(f"Exception details: {type(e).__name__}: {str(e)}")


class TotalTrainingTimeCallback(Callback):
    """Callback to monitor total training time."""

    def __init__(self) -> None:
        self.train_start_time = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.train_start_time = time.perf_counter_ns()
        rank_zero_info("Training started - timing initialized")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.train_start_time:
            total_time_ns = time.perf_counter_ns() - self.train_start_time
            total_time_seconds = total_time_ns / 1_000_000_000  # Convert ns to seconds
            hours = total_time_seconds // 3600
            minutes = (total_time_seconds % 3600) // 60
            seconds = total_time_seconds % 60
            trainer.logger.experiment.log({"total_training_time_seconds": total_time_seconds})
            rank_zero_info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
