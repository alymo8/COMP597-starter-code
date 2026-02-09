from src.config.util.base_config import _Arg, _BaseConfig

config_name = "simple"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(type=str, default=".", help="Directory where simple trainer stats files are written.")
        self._arg_output_file_prefix = _Arg(type=str, default="simple_stats", help="Filename prefix for simple trainer stats files.")
