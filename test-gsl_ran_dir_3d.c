/* 
  Tests gsl_ran_dir_3d function
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int main(int argc, char const *argv[])
{
	gsl_rng *r_global;


	return 0;
}


int seed=42;

r = gsl_rng_alloc(gsl_rng_mt19937);	/* use Mersenne twister */
//r = gsl_rng_alloc (gsl_rng_default);
gsl_rng_set(r, seed);

}


double
test_dir3dxy (void)
{
  double x = 0, y = 0, z = 0, theta;
  gsl_ran_dir_3d (r_global, &x, &y, &z);
  theta = atan2 (x, y);
  return theta;
}