from src.config.util.base_config import _Arg, _BaseConfig

config_name = "resource"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_log_every = _Arg(type=int, default=1, help="Log resource stats every N steps.")
        self._arg_include_gpu = _Arg(type=int, default=1, help="Enable GPU utilization and memory stats (1=yes, 0=no).")
        self._arg_include_system = _Arg(type=int, default=1, help="Enable system memory stats (1=yes, 0=no).")
        self._arg_include_process = _Arg(type=int, default=1, help="Enable process memory stats (1=yes, 0=no).")
        self._arg_include_cpu = _Arg(type=int, default=1, help="Enable CPU utilization stats (1=yes, 0=no).")
        self._arg_include_io = _Arg(type=int, default=1, help="Enable process I/O stats (1=yes, 0=no).")
        self._arg_include_energy = _Arg(type=int, default=1, help="Enable GPU power and energy tracking via NVML (1=yes, 0=no).")
        self._arg_include_torch_cuda_memory = _Arg(type=int, default=1, help="Enable torch CUDA allocated/reserved memory stats (1=yes, 0=no).")
        self._arg_carbon_intensity_gco2_per_kwh = _Arg(type=float, default=40.0, help="Carbon intensity for CO2 estimation (gCO2/kWh).")
        self._arg_gpu_index = _Arg(type=int, default=-1, help="GPU index to monitor (-1 uses device index or 0).")
        self._arg_output_dir = _Arg(type=str, default=".", help="Directory where resource stats files are written.")
        self._arg_output_file_prefix = _Arg(type=str, default="resource_stats", help="Filename prefix for resource stats files.")
