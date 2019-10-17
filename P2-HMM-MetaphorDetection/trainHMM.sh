#!/bin/bash

for l in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 3 4 5 
do
    for unk in 100
#0 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 
    do 
        echo 'lambda =' $l >> lambda_hmm.txt
        python3 HMM.py $l $unk
        python3 eval.py --pred val_output.csv >> lambda_hmm.txt
    done
done
