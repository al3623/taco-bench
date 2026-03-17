#!/bin/bash

for i in $(find ../multi_tensors -name *.tns); do 
		echo $i;
		./taco-bench -E=10 -i=B:$i > $i.txt
done
