import logging
import csv
import json
import os
import time
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
        self.include_cpu = bool(int(getattr(cfg, "include_cpu", 1)))
        self.include_io = bool(int(getattr(cfg, "include_io", 1)))
        self.include_energy = bool(int(getattr(cfg, "include_energy", 1)))
        self.include_torch_cuda_memory = bool(int(getattr(cfg, "include_torch_cuda_memory", 1)))
        self.carbon_intensity_gco2_per_kwh = float(getattr(cfg, "carbon_intensity_gco2_per_kwh", 40.0))
        self.gpu_index = int(getattr(cfg, "gpu_index", -1))
        self.output_dir = str(getattr(cfg, "output_dir", "."))
        self.output_file_prefix = str(getattr(cfg, "output_file_prefix", "resource_stats"))
        self.step_csv_path = os.path.join(self.output_dir, f"{self.output_file_prefix}_steps.csv")
        self.summary_json_path = os.path.join(self.output_dir, f"{self.output_file_prefix}_summary.json")

        # Stats containers
        self.gpu_util = utils.RunningStat()
        self.gpu_mem_used = utils.RunningStat()
        self.gpu_mem_total = utils.RunningStat()
        self.gpu_power_w = utils.RunningStat()
        self.gpu_energy_mj = utils.RunningStat()
        self.step_energy_kwh = utils.RunningStat()
        self.step_carbon_gco2 = utils.RunningStat()
        self.sys_mem_used = utils.RunningStat()
        self.sys_mem_total = utils.RunningStat()
        self.proc_rss = utils.RunningStat()
        self.proc_vms = utils.RunningStat()
        self.sys_cpu_percent = utils.RunningStat()
        self.proc_cpu_percent = utils.RunningStat()
        self.proc_io_read = utils.RunningStat()
        self.proc_io_write = utils.RunningStat()
        self.step_duration_ms = utils.RunningStat()
        self.torch_cuda_allocated_mib = utils.RunningStat()
        self.torch_cuda_reserved_mib = utils.RunningStat()

        self._process = None
        self._prev_io = None
        self._nvml_ready = False
        self._nvml_handle = None
        self._prev_total_energy_mj = None
        self._cumulative_energy_kwh = 0.0
        self._cumulative_carbon_gco2 = 0.0
        self._step_start_ts = None
        self._train_start_ts = None
        self._train_duration_sec = 0.0
        self._csv_file = None
        self._csv_writer = None
        self._step_fieldnames = [
            "step",
            "elapsed_sec",
            "step_duration_ms",
            "gpu_util",
            "gpu_mem_used",
            "gpu_mem_total",
            "gpu_power_w",
            "gpu_energy_mj",
            "step_energy_kwh",
            "step_carbon_gco2",
            "cumulative_gpu_energy_kwh",
            "cumulative_carbon_gco2",
            "torch_cuda_allocated_mib",
            "torch_cuda_reserved_mib",
            "sys_mem_used",
            "sys_mem_total",
            "sys_cpu_percent",
            "proc_rss",
            "proc_vms",
            "proc_cpu_percent",
            "io_read",
            "io_write",
        ]

    def start_train(self) -> None:
        self._train_start_ts = time.perf_counter()
        os.makedirs(self.output_dir, exist_ok=True)
        self._csv_file = open(self.step_csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._step_fieldnames)
        self._csv_writer.writeheader()

        if psutil is None:
            logger.warning("psutil is not available; system/process stats will be disabled")
            self.include_system = False
            self.include_process = False
            self.include_cpu = False
            self.include_io = False
        else:
            self._process = psutil.Process()
            if self.include_cpu:
                psutil.cpu_percent(interval=None)
                self._process.cpu_percent(interval=None)

        if self.include_gpu:
            if pynvml is None:
                logger.warning("pynvml is not available; GPU stats will be disabled")
                self.include_gpu = False
                self.include_energy = False
            else:
                try:
                    pynvml.nvmlInit()
                    self._nvml_ready = True
                    self._nvml_handle = self._get_nvml_handle()
                    if self.include_energy:
                        try:
                            self._prev_total_energy_mj = float(
                                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle)
                            )
                        except Exception:
                            logger.warning("NVML total energy counter unavailable; disabling energy and carbon tracking.")
                            self.include_energy = False
                except Exception:
                    logger.exception("Failed to initialize NVML; GPU stats will be disabled")
                    self.include_gpu = False
                    self.include_energy = False

    def stop_train(self) -> None:
        if self._train_start_ts is not None:
            self._train_duration_sec = time.perf_counter() - self._train_start_ts

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                logger.exception("Failed to shutdown NVML cleanly")

    def start_step(self) -> None:
        self._step_start_ts = time.perf_counter_ns()

    def stop_step(self) -> None:
        if self._step_start_ts is not None:
            delta_ms = (time.perf_counter_ns() - self._step_start_ts) / 1000000.0
            self.step_duration_ms.update(delta_ms)
            self._step_start_ts = None

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

        if self._csv_writer is not None:
            row = {key: sample.get(key, None) for key in self._step_fieldnames}
            self._csv_writer.writerow(row)
            if self._csv_file is not None:
                self._csv_file.flush()

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
        if "step_energy_kwh" in sample:
            parts.append(
                f"energy={sample['step_energy_kwh'] * 1000:.4f} Wh carbon={sample['step_carbon_gco2']:.4f} gCO2e"
            )
        print(" | ".join(parts))

    def log_stats(self) -> None:
        print("RESOURCE STATS (averages)")
        if self.gpu_util.history:
            self._print_avg("gpu_util", self.gpu_util, "%")
            self._print_avg("gpu_mem_used", self.gpu_mem_used, "MiB")
            self._print_avg("gpu_mem_total", self.gpu_mem_total, "MiB")
        if self.gpu_power_w.history:
            self._print_avg("gpu_power_w", self.gpu_power_w, "W")
        if self.gpu_energy_mj.history:
            self._print_avg("gpu_energy_mj", self.gpu_energy_mj, "mJ")
        if self.step_energy_kwh.history:
            self._print_avg("step_energy_kwh", self.step_energy_kwh, "kWh")
            self._print_avg("step_carbon_gco2", self.step_carbon_gco2, "gCO2e")
            print(f"  cumulative_gpu_energy_kwh: {self._cumulative_energy_kwh:.8f}")
            print(f"  cumulative_carbon_gco2: {self._cumulative_carbon_gco2:.6f}")
        if self.sys_mem_used.history:
            self._print_avg("sys_mem_used", self.sys_mem_used, "MiB")
            self._print_avg("sys_mem_total", self.sys_mem_total, "MiB")
        if self.proc_rss.history:
            self._print_avg("proc_rss", self.proc_rss, "MiB")
            self._print_avg("proc_vms", self.proc_vms, "MiB")
        if self.torch_cuda_allocated_mib.history:
            self._print_avg("torch_cuda_allocated_mib", self.torch_cuda_allocated_mib, "MiB")
            self._print_avg("torch_cuda_reserved_mib", self.torch_cuda_reserved_mib, "MiB")
        if self.proc_io_read.history:
            self._print_avg("io_read", self.proc_io_read, "MB")
            self._print_avg("io_write", self.proc_io_write, "MB")
        if self.step_duration_ms.history:
            self._print_avg("step_duration_ms", self.step_duration_ms, "ms")
        if self.sys_cpu_percent.history:
            self._print_avg("sys_cpu_percent", self.sys_cpu_percent, "%")
        if self.proc_cpu_percent.history:
            self._print_avg("proc_cpu_percent", self.proc_cpu_percent, "%")
        print(f"  total_training_time_sec: {self._train_duration_sec:.3f}")

        summary = {
            "iterations": self.iteration,
            "total_training_time_sec": self._train_duration_sec,
            "carbon_intensity_gco2_per_kwh": self.carbon_intensity_gco2_per_kwh,
            "cumulative_gpu_energy_kwh": self._cumulative_energy_kwh,
            "cumulative_carbon_gco2": self._cumulative_carbon_gco2,
            "averages": {
                "gpu_util": self._avg_or_none(self.gpu_util),
                "gpu_mem_used_mib": self._avg_or_none(self.gpu_mem_used),
                "gpu_mem_total_mib": self._avg_or_none(self.gpu_mem_total),
                "gpu_power_w": self._avg_or_none(self.gpu_power_w),
                "gpu_energy_mj": self._avg_or_none(self.gpu_energy_mj),
                "step_energy_kwh": self._avg_or_none(self.step_energy_kwh),
                "step_carbon_gco2": self._avg_or_none(self.step_carbon_gco2),
                "torch_cuda_allocated_mib": self._avg_or_none(self.torch_cuda_allocated_mib),
                "torch_cuda_reserved_mib": self._avg_or_none(self.torch_cuda_reserved_mib),
                "sys_mem_used_mib": self._avg_or_none(self.sys_mem_used),
                "sys_mem_total_mib": self._avg_or_none(self.sys_mem_total),
                "sys_cpu_percent": self._avg_or_none(self.sys_cpu_percent),
                "proc_rss_mib": self._avg_or_none(self.proc_rss),
                "proc_vms_mib": self._avg_or_none(self.proc_vms),
                "proc_cpu_percent": self._avg_or_none(self.proc_cpu_percent),
                "io_read_mb": self._avg_or_none(self.proc_io_read),
                "io_write_mb": self._avg_or_none(self.proc_io_write),
                "step_duration_ms": self._avg_or_none(self.step_duration_ms),
            },
            "peaks": {
                "gpu_util": self._max_or_none(self.gpu_util),
                "gpu_mem_used_mib": self._max_or_none(self.gpu_mem_used),
                "gpu_power_w": self._max_or_none(self.gpu_power_w),
                "torch_cuda_allocated_mib": self._max_or_none(self.torch_cuda_allocated_mib),
                "torch_cuda_reserved_mib": self._max_or_none(self.torch_cuda_reserved_mib),
                "sys_mem_used_mib": self._max_or_none(self.sys_mem_used),
                "proc_rss_mib": self._max_or_none(self.proc_rss),
                "step_duration_ms": self._max_or_none(self.step_duration_ms),
            },
        }
        with open(self.summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"resource summary saved to {self.summary_json_path}")

    def _print_avg(self, name: str, stat: utils.RunningStat, unit: str) -> None:
        avg = stat.get_average()
        print(f"  {name}: {avg:.3f} {unit}")

    def _avg_or_none(self, stat: utils.RunningStat) -> Optional[float]:
        if not stat.history:
            return None
        return float(stat.get_average())

    def _max_or_none(self, stat: utils.RunningStat) -> Optional[float]:
        if not stat.history:
            return None
        return float(max(stat.history))

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
        elapsed_sec = 0.0
        if self._train_start_ts is not None:
            elapsed_sec = time.perf_counter() - self._train_start_ts

        sample: Dict[str, float] = {
            "step": self.iteration,
            "elapsed_sec": elapsed_sec,
            "step_duration_ms": self.step_duration_ms.get_last(),
        }

        if self.include_gpu and self._nvml_ready and self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_util = float(util.gpu)
                gpu_mem_used = float(mem.used) / (1024 * 1024)
                gpu_mem_total = float(mem.total) / (1024 * 1024)
                try:
                    gpu_power_w = float(pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)) / 1000.0
                    self.gpu_power_w.update(gpu_power_w)
                    sample["gpu_power_w"] = gpu_power_w
                except Exception:
                    pass

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
                if self.include_energy:
                    try:
                        total_energy_mj = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle))
                        if self._prev_total_energy_mj is None:
                            step_energy_mj = 0.0
                        else:
                            step_energy_mj = max(0.0, total_energy_mj - self._prev_total_energy_mj)
                        self._prev_total_energy_mj = total_energy_mj

                        step_energy_kwh = step_energy_mj / 3_600_000_000.0
                        step_carbon_gco2 = step_energy_kwh * self.carbon_intensity_gco2_per_kwh
                        self._cumulative_energy_kwh += step_energy_kwh
                        self._cumulative_carbon_gco2 += step_carbon_gco2

                        self.gpu_energy_mj.update(step_energy_mj)
                        self.step_energy_kwh.update(step_energy_kwh)
                        self.step_carbon_gco2.update(step_carbon_gco2)
                        sample.update(
                            {
                                "gpu_energy_mj": step_energy_mj,
                                "step_energy_kwh": step_energy_kwh,
                                "step_carbon_gco2": step_carbon_gco2,
                                "cumulative_gpu_energy_kwh": self._cumulative_energy_kwh,
                                "cumulative_carbon_gco2": self._cumulative_carbon_gco2,
                            }
                        )
                    except Exception:
                        logger.warning("Failed to read NVML total energy counter; disabling energy and carbon tracking.")
                        self.include_energy = False
            except Exception:
                logger.exception("Failed to read GPU stats")
                self.include_gpu = False
                self.include_energy = False

        if (
            self.include_torch_cuda_memory
            and torch.cuda.is_available()
            and self.device is not None
            and getattr(self.device, "type", None) == "cuda"
        ):
            try:
                device_idx = self.device.index
                if device_idx is None:
                    device_idx = torch.cuda.current_device()
                allocated_mib = float(torch.cuda.memory_allocated(device_idx)) / (1024 * 1024)
                reserved_mib = float(torch.cuda.memory_reserved(device_idx)) / (1024 * 1024)
                self.torch_cuda_allocated_mib.update(allocated_mib)
                self.torch_cuda_reserved_mib.update(reserved_mib)
                sample["torch_cuda_allocated_mib"] = allocated_mib
                sample["torch_cuda_reserved_mib"] = reserved_mib
            except Exception:
                logger.warning("Failed to read torch CUDA memory stats; disabling torch CUDA memory tracking.")
                self.include_torch_cuda_memory = False

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
                if self.include_cpu:
                    sys_cpu = float(psutil.cpu_percent(interval=None))
                    self.sys_cpu_percent.update(sys_cpu)
                    sample["sys_cpu_percent"] = sys_cpu
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
                if self.include_cpu:
                    proc_cpu = float(self._process.cpu_percent(interval=None))
                    self.proc_cpu_percent.update(proc_cpu)
                    sample["proc_cpu_percent"] = proc_cpu
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

        if len(sample) <= 3:
            return None
        return sample
