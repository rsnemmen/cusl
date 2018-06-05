# compilation
gcc test-gsl_bessel_Kn.c -o test-gsl_bessel_Kn -lgsl -lm -lgslcblas
echo "Compiled GSL code"
nvcc test-cu_bessel_Kn.cu -o test-cu_bessel_Kn -I../src
echo "Compiled CUDA code"

# tests
./test-gsl_bessel_Kn 
./test-cu_bessel_Kn

# plot
python plot-bessel.py