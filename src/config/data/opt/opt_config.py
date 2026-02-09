from src.config.util.base_config import _Arg, _BaseConfig

config_name = "opt"


class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_vocab_size = _Arg(type=int, help="Vocabulary size used for synthetic token generation.", default=50272)
        self._arg_seq_len = _Arg(type=int, help="Sequence length for each generated sample.", default=1024)
        self._arg_dataset_gb = _Arg(type=float, help="Approximate synthetic dataset size in GiB before capping.", default=2.5)
        self._arg_batch_size = _Arg(type=int, help="Batch size for the OPT synthetic data loader.", default=1)
        self._arg_num_workers = _Arg(type=int, help="Number of data loader workers.", default=2)
        self._arg_max_samples = _Arg(type=int, help="Maximum number of samples to keep. Set <=0 to disable cap.", default=0)
