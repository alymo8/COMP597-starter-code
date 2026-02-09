import logging
import csv
import json
import os
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import torch

logger = logging.getLogger(__name__)

trainer_stats_name="simple"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to simple trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return SimpleTrainerStats(device=device, conf=conf)

class SimpleTrainerStats(base.TrainerStats):
    """Provides simple timing measurements of training.

    This class measures the time used by the training steps, forward passes, 
    backward passes and optimizer steps.

    Parameters
    ----------
    device
        The PyTorch device used for training. The asynchronous nature of CUDA 
        implementations means the Python code of a pass might complete before 
        the GPU is done executing the tasks. As such, this device is used to 
        synchronize on the CUDA stream to which the executions are issued.

    Attributes
    ----------
    device : torch.device
        The PyTorch device as provided to the constructor.
    step_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each training step.
    forward_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each forward pass.
    backward_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each backward pass.
    optimizer_step_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each optimizer step.

    Notes
    -----
        This is should only be used when training is done on a CUDA device. It 
        will fail otherwise. Moreover, if the training is not done on the 
        default stream of the device, the measurements will be unreliable as 
        synchronization is only done on the default stream.

    """

    def __init__(self, device : torch.device, conf: config.Config) -> None:
        super().__init__()
        self.device = device
        self.step_stats = utils.RunningTimer()
        self.forward_stats = utils.RunningTimer()
        self.backward_stats = utils.RunningTimer()
        self.optimizer_step_stats = utils.RunningTimer()
        self.save_checkpoint_stats = utils.RunningTimer()
        self.iteration = 0

        simple_cfg = conf.trainer_stats_configs.simple
        self.output_dir = str(getattr(simple_cfg, "output_dir", "."))
        self.output_file_prefix = str(getattr(simple_cfg, "output_file_prefix", "simple_stats"))
        self.step_csv_path = os.path.join(self.output_dir, f"{self.output_file_prefix}_steps.csv")
        self.summary_json_path = os.path.join(self.output_dir, f"{self.output_file_prefix}_summary.json")
        self._csv_file = None
        self._csv_writer = None

    def start_train(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._csv_file = open(self.step_csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "step",
                "step_ms",
                "forward_ms",
                "backward_ms",
                "optimizer_step_ms",
                "checkpoint_ms",
            ],
        )
        self._csv_writer.writeheader()

    def stop_train(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def start_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.step_stats.start()

    def stop_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.step_stats.stop()

    def start_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.optimizer_step_stats.start()

    def stop_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.optimizer_step_stats.stop()

    def start_forward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.forward_stats.start()

    def stop_forward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.forward_stats.stop()

    def start_backward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.backward_stats.start()

    def stop_backward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.backward_stats.stop()

    def start_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)
        self.save_checkpoint_stats.start()

    def stop_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)
        self.save_checkpoint_stats.stop()

    def log_step(self) -> None:
        """Log the previous step's time measurements.

        This will print the measured time of the previous step, its forward 
        pass, backward pass and optimizer step. All the measurements are in 
        milliseconds.

        """
        self.iteration += 1
        step_ms = self.step_stats.get_last() / 1000000
        forward_ms = self.forward_stats.get_last() / 1000000
        backward_ms = self.backward_stats.get_last() / 1000000
        optimizer_ms = self.optimizer_step_stats.get_last() / 1000000
        checkpoint_ms = self.save_checkpoint_stats.get_last() / 1000000
        print(f"step {step_ms} -- forward {forward_ms} -- backward {backward_ms} -- optimizer step {optimizer_ms}")
        if self._csv_writer is not None:
            self._csv_writer.writerow(
                {
                    "step": self.iteration,
                    "step_ms": step_ms,
                    "forward_ms": forward_ms,
                    "backward_ms": backward_ms,
                    "optimizer_step_ms": optimizer_ms,
                    "checkpoint_ms": checkpoint_ms,
                }
            )

    def log_stats(self) -> None:
        """Log basic statistics on the time measurements.

        This will print the average time of each step, each forward pass, each 
        backward pass and each optimizer step. Then it prints a breakdown for 
        each of those. All measurements are in milliseconds.

        """
        print(f"AVG : step {self.step_stats.get_average() / 1000000} -- forward {self.forward_stats.get_average() / 1000000} -- backward {self.backward_stats.get_average() / 1000000} -- optimizer step {self.optimizer_step_stats.get_average() / 1000000}")
        print("###############        Step        ###############")
        self.step_stats.log_analysis()
        print("###############      FORWARD       ###############")
        self.forward_stats.log_analysis()
        print("###############      BACKWARD      ###############")
        self.backward_stats.log_analysis()
        print("###############   OPTIMIZER STEP   ###############")
        self.optimizer_step_stats.log_analysis()
        # NOTE: (greta) commented out for now - not using checkpointing stats
        # print("###############   CHECKPOINTING    #################")
        # self.save_checkpoint_stats.log_analysis()
        summary = {
            "iterations": self.iteration,
            "averages_ms": {
                "step_ms": self.step_stats.get_average() / 1000000,
                "forward_ms": self.forward_stats.get_average() / 1000000,
                "backward_ms": self.backward_stats.get_average() / 1000000,
                "optimizer_step_ms": self.optimizer_step_stats.get_average() / 1000000,
                "checkpoint_ms": self.save_checkpoint_stats.get_average() / 1000000,
            },
        }
        with open(self.summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"simple summary saved to {self.summary_json_path}")

    def log_loss(self, loss : torch.Tensor) -> None:
        pass

