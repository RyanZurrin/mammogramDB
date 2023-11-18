#!/bin/bash
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera13
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=8000
#SBATCH -t 3-00:00
#SBATCH --error=/tmp/slurm_%u_%x-%j_%a.err
#SBATCH --output=/tmp/slurm_%u_%x-%j_%a.out


CONTAINER=58a38fc60eeb
OMAMA_CHECKOUT=/home/$USER/Projects/omama/

#
# SETUP ROOTLESS DOCKER
#
export HOME=/home2/$USER
export XDG_RUNTIME_DIR=$HOME/.docker/xrd
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

# RUN DOCKER DEMON ON DEMAND
if [[ "$(pgrep -u $USER rootless)" == "" ]]; then
  echo "*** starting Docker in $XDG_RUNTIME_DIR"
  rootlesskit rm -rf $XDG_RUNTIME_DIR
  mkdir -p $XDG_RUNTIME_DIR
  dockerd-rootless.sh 2> /tmp/docker_$USER.log &
  dockerPID=$!
fi

sleep 1

# INSTALL DEEPSIGHT, IF NOT THERE
if [[ "$(docker image inspect $CONTAINER --format='whoisthebest')" != "whoisthebest" ]]; then
  echo '*** installing deepsight.. (takes some time!) ***'
  docker load < /raid/mpsych/DeepSight-2.1.1.tar
fi

#
#
#
# PYTHON LAND!!!
#
#
#
. /home/$USER/.bashrc
conda activate O

# GO TO OMAMA REPO
cd $OMAMA_CHECKOUT

echo "*** Calling DeepSight API.."
python _SLURM/_TESTS/one2d.py
#
#
#
# LEAVE PYTHON LAND!!!
#
#
#



# SAYONARA!!
exit 0;

