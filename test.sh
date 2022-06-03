#!/usr/bin/env bash
 
method=NN

params=`cat input/hyperparameters_$method.json`
paramsLength=`echo $params | jq length`

for ((i=0; i<=`expr $paramsLength - 1`; i++)) do
    num_hidden_layer=`echo $params | jq -r ".[$i].num_hidden_layer"`
    hidden_feature=`echo $params | jq -r ".[$i].hidden_feature"`
    alpha=`echo $params | jq -r ".[$i].alpha"`
    type_odm=`echo $params | jq -r ".[$i].type_odm"`
    echo $num_hidden_layer
done
    

