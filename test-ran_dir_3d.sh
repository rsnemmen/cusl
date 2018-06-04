# compile 
gcc test-gsl_ran_dir_3d.c -o test-gsl_ran_dir_3d -lgsl -lm -lgslcblas
nvcc test-cu_ran_dir_3d.cu -o test-cu_ran_dir_3d

# test
./test-gsl_ran_dir_3d 10
echo
./test-cu_ran_dir_3d 10 1