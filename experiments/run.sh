#!/bin/bash
methods=("./methods/plnl")
data_dir="/nas/datasets"
datasets=("stl10")
distrs=(0)
nc=1

k=500
t=0.1

gpus=(3)
seed=999
index=0

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for distr in "${distrs[@]}"; do
            gpu=${gpus[$index]}                              # 选取当前 GPU
            index=$(( (index + 1) % ${#gpus[@]} ))
            file_path="./results/${method}_${dataset}_${distr}_${nc}_${k}_${t}_${seed}.log"
            log_dir=$(dirname "${file_path}")
            mkdir -p "${log_dir}"
            nohup python -u ${method}.py -dataset ${dataset} -data-dir ${data_dir}\
            -distr ${distr} -nc ${nc} -k ${k} -t ${t} \
            -gpu ${gpu} -seed ${seed} > ${file_path} 2>&1 &
        done
    done
done