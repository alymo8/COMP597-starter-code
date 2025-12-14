#!/bin/bash

script_dir=$(readlink -f -n $(dirname $0))

# Load config
. ${script_dir}/config.sh

# Set up admin directory

mkdir -p ${COMP597_ADMIN_DIR}
setfacl -m g:${COMP597_USERS_GROUP}:--- ${COMP597_ADMIN_DIR}

cd ${COMP597_ADMIN_DIR}

git clone ${COMP597_REPO_URL}

mkdir -p ${COMP597_PIP_CACHE}

if [[ ${COMP597_EMERGENCY_STORAGE_ENABLED} = true ]]; then

	mkdir -p ${COMP597_EMERGENCY_STORAGE_DIR}

	cd ${COMP597_EMERGENCY_STORAGE_DIR}

	for ((i=0; i<${COMP597_EMERGENCY_STORAGE_NB_BLOCKS}; i++)); do
		# Unfortunately, it seems that the nfs partition uses a version of nfs that is too old to use fallocate, which would be much faster.
		dd if=/dev/zero of=block-$i  bs=${COMP597_EMERGENCY_STORAGE_BLOCK_SIZE_IN_MB}  count=1
	done
fi

# Set up students' directory

mkdir -p ${COMP597_STUDENTS_DIR}
setfacl -m g:${COMP597_USERS_GROUP}:rwx ${COMP597_STUDENTS_DIR}
setfacl -d -m g:${COMP597_USERS_GROUP}:r-x ${COMP597_STUDENTS_DIR}
setfacl -d -m g:${COMP597_ADMIN_GROUP}:rwx ${COMP597_STUDENTS_DIR}

# Set up environment

mkdir -p ${COMP597_CONDA_DIR}
setfacl -m g:${COMP597_USERS_GROUP}:r-x ${COMP597_CONDA_DIR}
setfacl -d -m g:${COMP597_USERS_GROUP}:r-x ${COMP597_CONDA_DIR}
setfacl -m g:${COMP597_ADMIN_GROUP}:rwx ${COMP597_CONDA_DIR}
setfacl -d -m g:${COMP597_ADMIN_GROUP}:rwx ${COMP597_CONDA_DIR}

export PIP_CACHE_DIR=${COMP597_PIP_CACHE}

conda create --prefix=${COMP597_CONDA_ENV_PREFIX} python=${COMP597_CONDA_ENV_PYTHON_VERSION}

conda activate ${COMP597_CONDA_ENV_PREFIX}

requirements_file=${script_dir}/../requirements.txt

pip install -r ${requirements_file}

rm -rf ${COMP597_PIP_CACHE}/*
