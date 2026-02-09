import logging
from typing import Dict, Optional

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import torch

logger = logging.getLogger(__name__)

trainer_stats_name = "resource"

try:
    import psutil
except Exception:  # pragma: no cover - environment dependent
    psutil = None

try:
    import pynvml
except Exception:  # pragma: no cover - environment dependent
    pynvml = None


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning("No device provided to resource trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return ResourceTrainerStats(device=device, conf=conf)


class ResourceTrainerStats(base.TrainerStats):
    """Collects basic resource utilization statistics during training."""

    def __init__(self, device: torch.device, conf: config.Config) -> None:
        super().__init__()
        self.device = device
        self.conf = conf
        self.iteration = 0

        cfg = conf.trainer_stats_configs.resource
        self.log_every = max(int(getattr(cfg, "log_every", 1)), 1)
        self.include_gpu = bool(int(getattr(cfg, "include_gpu", 1)))
        self.include_system = bool(int(getattr(cfg, "include_system", 1)))
        self.include_process = bool(int(getattr(cfg, "include_process", 1)))
        self.include_io = bool(int(getattr(cfg, "include_io", 1)))
        self.gpu_index = int(getattr(cfg, "gpu_index", -1))

        # Stats containers
        self.gpu_util = utils.RunningStat()
        self.gpu_mem_used = utils.RunningStat()
        self.gpu_mem_total = utils.RunningStat()
        self.sys_mem_used = utils.RunningStat()
        self.sys_mem_total = utils.RunningStat()
        self.proc_rss = utils.RunningStat()
        self.proc_vms = utils.RunningStat()
        self.proc_io_read = utils.RunningStat()
        self.proc_io_write = utils.RunningStat()

        self._process = None
        self._prev_io = None
        self._nvml_ready = False
        self._nvml_handle = None

    def start_train(self) -> None:
        if psutil is None:
            logger.warning("psutil is not available; system/process stats will be disabled")
            self.include_system = False
            self.include_process = False
            self.include_io = False
        else:
            self._process = psutil.Process()

        if self.include_gpu:
            if pynvml is None:
                logger.warning("pynvml is not available; GPU stats will be disabled")
                self.include_gpu = False
            else:
                try:
                    pynvml.nvmlInit()
                    self._nvml_ready = True
                    self._nvml_handle = self._get_nvml_handle()
                except Exception:
                    logger.exception("Failed to initialize NVML; GPU stats will be disabled")
                    self.include_gpu = False

    def stop_train(self) -> None:
        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                logger.exception("Failed to shutdown NVML cleanly")

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        self.iteration += 1
        sample = self._sample()
        if sample is None:
            return
        if (self.iteration % self.log_every) != 0:
            return

        parts = [f"resource step {self.iteration}"]
        if "gpu_util" in sample:
            parts.append(
                f"gpu_util={sample['gpu_util']:.1f}% gpu_mem={sample['gpu_mem_used']:.0f}/{sample['gpu_mem_total']:.0f} MiB"
            )
        if "sys_mem_used" in sample:
            parts.append(
                f"sys_mem={sample['sys_mem_used']:.0f}/{sample['sys_mem_total']:.0f} MiB"
            )
        if "proc_rss" in sample:
            parts.append(
                f"proc_rss={sample['proc_rss']:.0f} MiB proc_vms={sample['proc_vms']:.0f} MiB"
            )
        if "io_read" in sample:
            parts.append(
                f"io=read {sample['io_read']:.2f} MB write {sample['io_write']:.2f} MB"
            )
        print(" | ".join(parts))

    def log_stats(self) -> None:
        print("RESOURCE STATS (averages)")
        if self.gpu_util.history:
            self._print_avg("gpu_util", self.gpu_util, "%")
            self._print_avg("gpu_mem_used", self.gpu_mem_used, "MiB")
            self._print_avg("gpu_mem_total", self.gpu_mem_total, "MiB")
        if self.sys_mem_used.history:
            self._print_avg("sys_mem_used", self.sys_mem_used, "MiB")
            self._print_avg("sys_mem_total", self.sys_mem_total, "MiB")
        if self.proc_rss.history:
            self._print_avg("proc_rss", self.proc_rss, "MiB")
            self._print_avg("proc_vms", self.proc_vms, "MiB")
        if self.proc_io_read.history:
            self._print_avg("io_read", self.proc_io_read, "MB")
            self._print_avg("io_write", self.proc_io_write, "MB")

    def _print_avg(self, name: str, stat: utils.RunningStat, unit: str) -> None:
        avg = stat.get_average()
        print(f"  {name}: {avg:.3f} {unit}")

    def _get_nvml_handle(self):
        if not self._nvml_ready:
            return None
        if self.gpu_index >= 0:
            return pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        if self.device is not None and getattr(self.device, "type", None) == "cuda":
            idx = getattr(self.device, "index", None)
            if idx is not None:
                return pynvml.nvmlDeviceGetHandleByIndex(idx)
        return pynvml.nvmlDeviceGetHandleByIndex(0)

    def _sample(self) -> Optional[Dict[str, float]]:
        sample: Dict[str, float] = {}

        if self.include_gpu and self._nvml_ready and self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_util = float(util.gpu)
                gpu_mem_used = float(mem.used) / (1024 * 1024)
                gpu_mem_total = float(mem.total) / (1024 * 1024)
                self.gpu_util.update(gpu_util)
                self.gpu_mem_used.update(gpu_mem_used)
                self.gpu_mem_total.update(gpu_mem_total)
                sample.update(
                    {
                        "gpu_util": gpu_util,
                        "gpu_mem_used": gpu_mem_used,
                        "gpu_mem_total": gpu_mem_total,
                    }
                )
            except Exception:
                logger.exception("Failed to read GPU stats")
                self.include_gpu = False

        if self.include_system and self._process is not None:
            try:
                vm = psutil.virtual_memory()
                sys_mem_used = float(vm.used) / (1024 * 1024)
                sys_mem_total = float(vm.total) / (1024 * 1024)
                self.sys_mem_used.update(sys_mem_used)
                self.sys_mem_total.update(sys_mem_total)
                sample.update(
                    {
                        "sys_mem_used": sys_mem_used,
                        "sys_mem_total": sys_mem_total,
                    }
                )
            except Exception:
                logger.exception("Failed to read system memory stats")
                self.include_system = False

        if self.include_process and self._process is not None:
            try:
                mem = self._process.memory_info()
                proc_rss = float(mem.rss) / (1024 * 1024)
                proc_vms = float(mem.vms) / (1024 * 1024)
                self.proc_rss.update(proc_rss)
                self.proc_vms.update(proc_vms)
                sample.update(
                    {
                        "proc_rss": proc_rss,
                        "proc_vms": proc_vms,
                    }
                )
            except Exception:
                logger.exception("Failed to read process memory stats")
                self.include_process = False

        if self.include_io and self._process is not None:
            try:
                io = self._process.io_counters()
                if self._prev_io is None:
                    delta_read = 0
                    delta_write = 0
                else:
                    delta_read = io.read_bytes - self._prev_io.read_bytes
                    delta_write = io.write_bytes - self._prev_io.write_bytes
                self._prev_io = io

                io_read_mb = float(delta_read) / (1024 * 1024)
                io_write_mb = float(delta_write) / (1024 * 1024)
                self.proc_io_read.update(io_read_mb)
                self.proc_io_write.update(io_write_mb)
                sample.update(
                    {
                        "io_read": io_read_mb,
                        "io_write": io_write_mb,
                    }
                )
            except Exception:
                logger.exception("Failed to read process I/O stats")
                self.include_io = False

        if len(sample) == 0:
            return None
        return sample
