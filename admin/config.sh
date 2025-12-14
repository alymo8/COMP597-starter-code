
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "This is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

COMP597_=""
# Storage directories
COMP597_STORAGE_DIR=/home/slurm/comp597
COMP597_CONDA_DIR=${COMP597_STORAGE_DIR}/conda
COMP597_CONDA_ENV_DIR=${COMP597_CONDA_DIR}/envs
COMP597_ADMIN_DIR=${COMP597_STORAGE_DIR}/admin
COMP597_EMERGENCY_STORAGE_DIR=${COMP597_ADMIN_DIR}/emergency-storage
COMP597_CACHE_DIR=${COMP597_ADMIN_DIR}/.cache
COMP597_PIP_CACHE=${COMP597_CACHE_DIR}/pip
COMP597_STUDENTS_DIR=${COMP597_STORAGE_DIR}/students

# Info
COMP597_REPO_URL="git@github.com:OMichaud0/COMP597-starter-code.git"
COMP597_ADMIN_GROUP="cs597-admins"
COMP597_USERS_GROUP="cs597-users"
COMP597_CONDA_ENV_NAME="comp597"
COMP597_CONDA_ENV_PREFIX=${COMP597_CONDA_ENV_DIR}/${COMP597_CONDA_ENV_NAME}
COMP597_CONDA_ENV_PYTHON_VERSION="3.14"
COMP597_EMERGENCY_STORAGE_ENABLED=false # Allows the creation of empty files to occupy some storage so if the partition gets full, we can delete them to do management without having to delete everything.
COMP597_EMERGENCY_STORAGE_NB_BLOCKS=5
COMP597_EMERGENCY_STORAGE_BLOCK_SIZE_IN_MB=500MB
