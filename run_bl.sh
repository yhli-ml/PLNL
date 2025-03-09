#!/bin/bash
# run baseline
method='scl-log'

dataset='cifar100'
distr=0
nc=50

lr=0.1 # 0.01 for stl
gpu=3
seed=1
bs=64

time=$(date +"%Y-%m-%d %H:%M:%S")

if [ ${distr} -eq 1 ]; then
    file_path="./results/base/${method}/${dataset}_${distr}_${seed}.log"
    log_dir=$(dirname "${file_path}")
    mkdir -p "${log_dir}"
    nohup python -u ${method}.py -dataset ${dataset} -distr ${distr} -lr ${lr} -gpu ${gpu} -bs ${bs} -seed ${seed} > ${file_path} 2>&1 &
else
    file_path="./results/base/${method}/${dataset}_${distr}_${nc}_${seed}.log"
    log_dir=$(dirname "${file_path}")
    mkdir -p "${log_dir}"
    nohup python -u ${method}.py -dataset ${dataset} -distr ${distr} -nc ${nc} -lr ${lr} -gpu ${gpu} -bs ${bs} -seed ${seed} > ${file_path} 2>&1 &
fi