import datasets
import src.config as config
import torch.utils.data

def load_data(conf : config.Config) -> torch.utils.data.Dataset:
    """Simple function to load a dataset based on the provided config object.
    """
    train_files = None
    if conf.dataset_train_files is not None and conf.dataset_train_files != "":
        train_files = {"train": conf.dataset_train_files}
    return datasets.load_dataset(conf.dataset, data_files=train_files, split=conf.dataset_split, num_proc=conf.dataset_load_num_proc)
