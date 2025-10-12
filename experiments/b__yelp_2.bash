#!/bin/bash

seed_list=(3407)
alpha_list=(  0.025)
kappa_list=( 0.01 0.05)
tau_list=(  0.2 0.5 0.25 0.15)
eta_list=(0.05 )
num_list=(   20 )
num_heads_list=(1)
num_cluster_list=(20  12   24  10 15  14 )
rate_list=(0.75  0.85  0.725  0.8 0.825  )
ccal_list=(0.02   0.001  )
cluster_num_list=(  20   15  12  )

dataset="yelp"
gpu_id=1

# Full grid search over hyperparameters
for seed in "${seed_list[@]}"; do
  for alpha in "${alpha_list[@]}"; do
    for kappa in "${kappa_list[@]}"; do
      for tau in "${tau_list[@]}"; do
        for eta in "${eta_list[@]}"; do
          for num in "${num_list[@]}"; do
            for num_heads in "${num_heads_list[@]}"; do
              for num_cluster in "${num_cluster_list[@]}"; do
                for rate in "${rate_list[@]}"; do
                  for ccal in "${ccal_list[@]}"; do
                    for cluster_num in "${cluster_num_list[@]}"; do

                      echo "Running with seed=$seed, alpha=$alpha, kappa=$kappa, tau=$tau, eta=$eta, num=$num, num_heads=$num_heads, num_cluster=$num_cluster, cluster_num=$cluster_num, rate=$rate, ccal=$ccal"

                      python main.py \
                        --dataset $dataset \
                        --gpu_id $gpu_id \
                        --alpha $alpha \
                        --kappa $kappa \
                        --tau $tau \
                        --eta $eta \
                        --num $num \
                        --num_heads $num_heads \
                        --num_cluster $num_cluster \
                        --cluster_num $cluster_num \
                        --rate $rate \
                        --ccal $ccal \
                        --check_path "" \
                        --num_train_epochs 200 \
                        --patience 20 \
                        --ts_user 12 \
                        --ts_item 13 \
                        --freeze \
                        --log \
                        --user_sim_func cl \
                        --use_cross_att \
                        --seed $seed \
                        --num_workers 8

                      echo "---------------------------------"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
