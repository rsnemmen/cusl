/* 
  Tests gsl_ran_dir_3d function
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define SEED 42

void test_dir3d(gsl_rng *r, int n);





int main(int argc, char const *argv[])
{
	int n;
	gsl_rng *r;

    // handle command-line argument
    if ( argc != 2 ) {
        printf( "usage: %s <number of randoms desired> \n", argv[0] );
        exit(0);
    }  
    // n = number of random numbers (and threads)
    sscanf(argv[1], "%d", &n); 

	// define RNG type
	//r = gsl_rng_alloc(gsl_rng_mt19937);	/* use Mersenne twister */
	r = gsl_rng_alloc(gsl_rng_default);

	// initialize RNG
	gsl_rng_set(r, SEED);

	// generates n random numbers
	printf("Generating RNs on the CPU with GSL:\n");
	test_dir3d(r, n);

	return 0;
}




/*
  n = number of random vectors desired
*/
void test_dir3d(gsl_rng *r, int n)
{
	double x = 0, y = 0, z = 0, norm;

	for (int i=0; i<n; i++) {
		gsl_ran_dir_3d(r, &x, &y, &z);
		norm=sqrt(x*x+y*y+z*z);
		printf("i=%d x=%f y=%f z=%f norm=%f\n", i, x, y, z, norm);

		//theta = atan2(x, y);
	}
}