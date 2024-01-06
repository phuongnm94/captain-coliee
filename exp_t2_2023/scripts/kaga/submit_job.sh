#!/bin/bash

TYPE=${1:-"hp"}

if [[ $TYPE = "hp" ]];
then
    bash -c "qsub -M thanh.ptit.96@gmail.com -m be ./scripts/kaga/job_hp_search.sh"
fi

if [[ $TYPE = "ens" ]];
then
    bash -c "qsub -M thanh.ptit.96@gmail.com -m be ./scripts/kaga/job_ens.sh"
fi