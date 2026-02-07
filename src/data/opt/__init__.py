import torch
from torch.utils.data import Dataset, DataLoader

# This name is picked up by the auto-discovery system in src/data/__init__.py.
data_load_name = "opt"

BYTES_PER_TOKEN = 24  # input_ids + attention_mask + labels (int64 each => 3*8)


class SyntheticCausalLMDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, target_gb: float):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        target_bytes = int(target_gb * 1024**3)
        total_tokens = max(1, target_bytes // BYTES_PER_TOKEN)
        self.num_samples = max(1, total_tokens // seq_len)

        self.data = torch.randint(
            0, vocab_size, (self.num_samples, seq_len), dtype=torch.long
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_data(conf):
    # Prefer putting these under conf.data_configs.opt.*, but tolerate flat args too.
    opt_cfg = getattr(getattr(conf, "data_configs", None), "opt", None)

    def pick(name, default):
        if opt_cfg is not None and hasattr(opt_cfg, name):
            return getattr(opt_cfg, name)
        return getattr(conf, name, default)

    vocab_size = pick("vocab_size", 50272)     # OPT default vocab size
    seq_len = pick("seq_len", 1024)
    dataset_gb = pick("dataset_gb", 2.5)
    batch_size = pick("batch_size", 1)
    num_workers = pick("num_workers", 2)

    dataset = SyntheticCausalLMDataset(vocab_size, seq_len, dataset_gb)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
