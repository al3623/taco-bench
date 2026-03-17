#!/bin/bash

# TTV
echo "Running TTV"
./taco-bench -E=7 -r=1 -i=B:../tensors/nell-2.tns > ttv.txt

echo "Running MTTKRP"
# MTTKRP
./taco-bench -E=9 -r=1 -i=B:../tensors/nell-2.tns > mttkrp.txt
