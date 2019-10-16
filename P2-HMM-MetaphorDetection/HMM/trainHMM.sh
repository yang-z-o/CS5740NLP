#!/bin/bash

for k in 0 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8 25.6 51.2 102.4
do
    python3 HMM.py $k
    python3 eval.py --pred val_output.csv
done