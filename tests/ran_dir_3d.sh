#!/bin/sh
set -e

# compile 
gcc ran_dir_3d-gsl.c -o ran_dir_3d-gsl -lgsl -lm -lgslcblas
echo "Compiled GSL code"
nvcc ran_dir_3d.cu -o ran_dir_3d -I../src -lcurand
echo "Compiled CUDA code"

# test
./ran_dir_3d-gsl 10
echo
./ran_dir_3d 10 1
