#!/usr/bin/env bash

for dataset in synthetic_consensus synthetic_clustering synthetic_polarization #sample_twitter_Abortion
do
    #for method in Voter DeGroot AsLM SLANT SLANT+ NN SINN
    for method in NN SINN 
    do
        if [ $method=="SINN" ]; then
            type_odms=(DeGroot BCM SBCM FJ)
        else
            type_odms=(DeGroot)
        fi

        for type_odm in "${type_odms[@]}" 
        do
            echo $type_odm
            python3 main_sinn.py \
              --method $method \
              --dataset $dataset \
              --save_dir output/ \
              --num_hidden_layers 5 \
              --hidden_features 8 \
              --alpha 0.1 \
              --beta 0.1 \
              --num_epochs 500 \
              --lr 0.001 \
              --K 1 \
              --type_odm $type_odm 
        done
    done
done
