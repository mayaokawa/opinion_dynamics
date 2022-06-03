#!/usr/bin/env bash

for dataset in synthetic_consensus synthetic_clustering synthetic_polarization #sample_twitter_Abortion
do
    for method in NN SINN
    do
        ## Load hyperparameters from configuration file
        params=`cat input/hyperparameters_$method.json`
        paramsLength=`echo $params | jq length`
    
        ## Loop over different set of hyperparameters
        for ((i=0; i<=`expr $paramsLength - 1`; i++)) do
            num_hidden_layer=`echo $params | jq -r ".[$i].num_hidden_layer"`
            hidden_feature=`echo $params | jq -r ".[$i].hidden_feature"`
            alpha=`echo $params | jq -r ".[$i].alpha"`
            type_odm=`echo $params | jq -r ".[$i].type_odm"`

            ## Run main function at each iteration of loop 
            python3 main_sinn.py \
                  --method $method \
                  --dataset $dataset \
                  --save_dir output/ \
                  --num_hidden_layers $num_hidden_layer \
                  --hidden_features $hidden_feature \
                  --alpha $alpha \
                  --beta 0.1 \
                  --num_epochs 500 \
                  --lr 0.001 \
                  --K 1 \
                  --type_odm $type_odm
        done
    done
done

