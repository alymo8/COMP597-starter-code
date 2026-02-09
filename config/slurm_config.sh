if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

# Override SLURM defaults for this repo.
export COMP597_SLURM_TIME_LIMIT="200:00"
