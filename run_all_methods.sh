#!/usr/bin/env bash

for method in Voter DeGroot AsLM SLANT SLANT+ NN Propose
do
    for dataset in synthetic_consensus synthetic_clustering synthetic_polarization #sample_twitter_Abortion
    do
        python3 main_sinn.py \
          --method $method \
          --dataset $dataset \
          --save_dir output/ \
          --num_hidden_layers 7 \
          --hidden_features 12 \
          --alpha 0.1 \
          --beta 0.1 \
          --num_epochs 1 \
          --lr 0.0001 \
          --K 1 \
          --type_odm DeGroot 
    done
done
