from src.config.util.base_config import _Arg, _BaseConfig

config_name="codecarbon"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(type=int, help="The run number used for codecarbon file tracking.", default=0)
        self._arg_project_name = _Arg(type=str, help="The name of the project used for codecarbon file tracking.", default="energy-efficiency")
        self._arg_output_dir = _Arg(type=str, help="The path of the output directory where files will be saved.", default=".")
        self._arg_track_steps = _Arg(type=int, help="Track step-level tasks (1=yes, 0=no).", default=1)
        self._arg_track_substeps = _Arg(type=int, help="Track phase-level tasks (1=yes, 0=no).", default=1)

