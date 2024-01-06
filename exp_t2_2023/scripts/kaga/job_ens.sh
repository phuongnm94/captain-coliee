#!/bin/bash

#PBS -q GPU-1
#PBS -oe
#PBS -l select=1
#PBS -N 2210421-monot5-hp

cd /home/s2210421/projects/coliee/monoT5

module load singularity/3.9.5

singularity exec --nv /home/s2210421/docker_images/tensorflow-notebook.sif ./scripts/run_ens.sh