#!/usr/bin/env bash

for dataset in synthetic_consensus #synthetic_clustering synthetic_polarization #sample_twitter_Abortion
do
    #for method in Voter DeGroot AsLM SLANT SLANT+ NN SINN
    for method in SINN 
    do
        python3 main_sinn.py \
              --method $method \
              --dataset $dataset \
              --save_dir output/ \
              --num_hidden_layers 5 \
              --hidden_features 12 \
              --alpha 0.1 \
              --beta 0.1 \
              --num_epochs 500 \
              --lr 0.001 \
              --K 1 \
              --type_odm SBCM 
    done
done
