import math

import torch
from torch.utils.data import DataLoader, Dataset

# This name is picked up by the auto-discovery system in src/data/__init__.py.
data_load_name = "opt"

BYTES_PER_TOKEN = 24  # input_ids + attention_mask + labels (int64 each => 3*8)

generators = {}


def register_generator(fn):
    generators[fn.__name__.lstrip("gen_")] = fn
    return fn


class SyntheticData(Dataset):
    def __init__(self, generator_fn, info, n: int, repeat: int):
        self.n = max(1, int(n))
        self.repeat = max(1, int(repeat))
        self.generator_fn = generator_fn
        self.info = info

    def __getitem__(self, i):
        return self.generator_fn(self.info)

    def __len__(self):
        return self.n * self.repeat


class _SyntheticInfo:
    def __init__(self, vocab_size: int, train_length: int):
        self.train_length = train_length
        self.vocab_size = vocab_size


def vocabgen(info):
    return torch.randint(0, info.vocab_size, (info.train_length,), dtype=torch.long)


def maskgen(info):
    return torch.ones((info.train_length,), dtype=torch.long)


@register_generator
def gen_AutoModelForCausalLM(info):
    input_ids = vocabgen(info)
    return {
        "input_ids": input_ids,
        "attention_mask": maskgen(info),
        "labels": input_ids.clone(),
    }


@register_generator
def gen_AutoModelForSeq2SeqLM(info):
    return gen_AutoModelForCausalLM(info)


@register_generator
def gen_AutoModelForMaskedLM(info):
    return gen_AutoModelForCausalLM(info)


def load_data(conf):
    # Prefer putting these under conf.data_configs.opt.*, but tolerate flat args too.
    opt_cfg = getattr(getattr(conf, "data_configs", None), "opt", None)

    def pick(name, default):
        if opt_cfg is not None and hasattr(opt_cfg, name):
            return getattr(opt_cfg, name)
        return getattr(conf, name, default)

    vocab_size = pick("vocab_size", 50272)  # OPT default vocab size
    seq_len = pick("seq_len", 1024)
    dataset_gb = pick("dataset_gb", 2.5)
    batch_size = pick("batch_size", 1)
    num_workers = pick("num_workers", 2)
    max_samples = pick("max_samples", 0)
    cache_samples = pick("cache_samples", 0)
    generator_name = pick("generator_name", "AutoModelForCausalLM")

    target_bytes = int(dataset_gb * 1024**3)
    total_tokens = max(1, target_bytes // BYTES_PER_TOKEN)
    total_samples = max(1, total_tokens // seq_len)
    if max_samples is not None and max_samples > 0:
        total_samples = min(total_samples, max_samples)

    if cache_samples is not None and cache_samples > 0:
        n = min(total_samples, cache_samples)
        repeat = max(1, math.ceil(total_samples / n))
    else:
        n = total_samples
        repeat = 1

    info = _SyntheticInfo(vocab_size=vocab_size, train_length=seq_len)
    generator_fn = generators.get(generator_name, gen_AutoModelForCausalLM)
    dataset = SyntheticData(generator_fn, info=info, n=n, repeat=repeat)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
