#!/bin/bash

for i in $(find ../multi_matrices -name *.mtx); do 
		echo $i;
		./taco-bench -E=0 -i=A:$i > $i.txt
done
