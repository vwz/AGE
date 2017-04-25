#!/bin/sh

i=0
while [ "$i" -lt 200 ] # while i < 1000
do
    python train_entropy_density_graphcentral_ts.py 0 4 6 citeseer
    echo $i
    i=`expr $i + 1`
done
