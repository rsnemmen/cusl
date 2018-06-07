# compile 
gcc test-gsl_ran_dir_3d.c -o test-gsl_ran_dir_3d -lgsl -lm -lgslcblas
echo "Compiled GSL code"
nvcc test-cu_ran_dir_3d.cu -o test-cu_ran_dir_3d -I../src
echo "Compiled CUDA code"

# test
./test-gsl_ran_dir_3d 10
echo
./test-cu_ran_dir_3d 10 1