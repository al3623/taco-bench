#!/bin/bash


./taco-bench -E=1 -r=100 -i=A:$1 > $1.txt

python out_to_csv.py $1.txt
