#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH -A kurs2014-1-124

increment=32
n=64
maximum=1024

while [[ $n -le $maximum ]]
do

./a.out $n $n >>gol.csv

n=$(( $n+$increment ))

done

