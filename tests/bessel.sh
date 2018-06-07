#!/bin/sh
#
# This script compiles and executes the testing units for the GSL and CUDA
# Bessel function implementations. After running this, you should have
# two files in the current folder: 
# 	• `bessel-gsl.dat`, output from GSL routines
# 	• `bessel-cuda.dat`, output from the CUDA routines
# 

# compilation
gcc bessel-gsl.c -o bessel-gsl -lgsl -lm -lgslcblas
echo "Compiled GSL code"
nvcc bessel-cusl.cu -o bessel-cusl -I../src
echo "Compiled CUDA code, new Bessel functions"
nvcc bessel-std.cu -o bessel-cuda-std -I../src
echo "Compiled CUDA code, default Bessel functions"
echo

# tests
echo "Running tests"
./bessel-gsl
./bessel-cusl
./bessel-cuda-std
echo

# plot
#python bessel-plot.py
python bessel-cusl-plot.py