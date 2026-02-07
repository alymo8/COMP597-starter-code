from typing import Any, Dict, Optional, Tuple

import torch
import torch.optim as optim
import torch.utils.data as data
import transformers

import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats

from .opt import load_opt

model_name = "opt"


def init_model(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    """
    Auto-discovery entrypoint for the OPT model.
    Returns (Trainer instance, optional kwargs dict passed to Trainer.train()).
    """
    opt_cfg = getattr(getattr(conf, "model_configs", None), "opt", None)
    hf_name = getattr(opt_cfg, "hf_name", "facebook/opt-350m") if opt_cfg else "facebook/opt-350m"
    dtype = getattr(opt_cfg, "dtype", "fp16") if opt_cfg else "fp16"

    model, tokenizer = load_opt(model_name=hf_name, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # If the data loader was already built (from src/data/opt), reuse it; otherwise wrap the dataset.
    if isinstance(dataset, data.DataLoader):
        loader = dataset
    else:
        loader = data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate)
    scheduler = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )

    # Trainer stats can be spelled "no-op"/"noop"; handle locally to avoid touching global trainer code.
    try:
        stats = trainer_stats.init_from_conf(conf=conf, device=device, num_train_steps=len(loader))
    except Exception as e:
        normalized = (
            str(conf.trainer_stats).strip().lower().replace("-", "").replace("_", "")
        )
        if normalized == "noop":
            from src.trainer.stats.noop import NOOPTrainerStats

            stats = NOOPTrainerStats()
        else:
            raise e
    if conf.trainer != "simple":
        raise Exception(f"Unknown trainer type {conf.trainer} for OPT")

    simple_trainer = trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=stats,
        conf=conf,
    )

    return simple_trainer, {"tokenizer": tokenizer}
