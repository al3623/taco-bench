#!/bin/bash


./taco-bench -E=1 -r=1 -i=A:$1 > $1.txt

# python3 out_to_csv.py $1.txt
