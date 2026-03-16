#!/bin/bash

for i in $(find ../matrices -name *.mtx); do 
		echo $i;
		./run_matrix.sh $i; 
done
