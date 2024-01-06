#!/bin/bash

#PBS -q GPU-1
#PBS -oe
#PBS -l select=1
#PBS -N 2210421-monot5-hp

cd /home/s2210421/projects/llms_for_legal

./scripts/kaga/run_hp_search.sh