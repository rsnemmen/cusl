#!/bin/sh
#
# This script compiles and executes the testing units for the GSL and CUDA
# Bessel function implementations. After running this, you should have
# two files in the current folder: 
# 	• `bessel-gsl.dat`, output from GSL routines
# 	• `bessel-cuda.dat`, output from the CUDA routines
# 

# compilation
gcc test-gsl_bessel.c -o test-gsl_bessel -lgsl -lm -lgslcblas
echo "Compiled GSL code"
nvcc test-cu_bessel.cu -o test-cu_bessel -I../src
echo "Compiled CUDA code"
echo

# tests
./test-gsl_bessel
./test-cu_bessel
echo

# plot
python plot-bessel.py