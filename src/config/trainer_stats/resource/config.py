from src.config.util.base_config import _Arg, _BaseConfig

config_name = "resource"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_log_every = _Arg(type=int, default=1, help="Log resource stats every N steps.")
        self._arg_include_gpu = _Arg(type=int, default=1, help="Enable GPU utilization and memory stats (1=yes, 0=no).")
        self._arg_include_system = _Arg(type=int, default=1, help="Enable system memory stats (1=yes, 0=no).")
        self._arg_include_process = _Arg(type=int, default=1, help="Enable process memory stats (1=yes, 0=no).")
        self._arg_include_io = _Arg(type=int, default=1, help="Enable process I/O stats (1=yes, 0=no).")
        self._arg_gpu_index = _Arg(type=int, default=-1, help="GPU index to monitor (-1 uses device index or 0).")
