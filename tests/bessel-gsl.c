/* 
  Compute all sorts of Bessel functions using the GSL Library
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_bessel.h>


#define N 100
#define x0 0.1
#define x1 5.
#define filename "bessel-gsl.dat"

int main(int argc, char const *argv[])
{
	int i;
	gsl_sf_result r;
	double *x = (double *)malloc(N*sizeof(double));
	double *K0 = (double *)malloc(N*sizeof(double));	
	double *K1 = (double *)malloc(N*sizeof(double));	
	// double *Kn_small = (double *)malloc(N*sizeof(double));	
	// double *Knu_e = (double *)malloc(N*sizeof(double));	
	// double *Knu_unife = (double *)malloc(N*sizeof(double));	
	double *Kn = (double *)malloc(N*sizeof(double));	

	for (i=0; i<N; i++) {
		// generate array of x-values
		x[i]=x0+i*(x1-x0)/(N-1);

		// scaled irregular modified cylindrical Bessel function of zeroth order \exp(x) K_0(x)
		gsl_sf_bessel_K0_scaled_e(x[i], &r);
		K0[i]=r.val;

		// irregular modified cylindrical Bessel function of first order, K_1(x), for x > 0. 
		gsl_sf_bessel_K1_scaled_e(x[i], &r);
		K1[i]=r.val;

		// gsl_bessel_Kn_scaled_small_x(2, x[i], &r);
		// Kn_small[i]=r.val;

		// gsl_sf_bessel_Knu_scaled_asympx_e(2.,x[i], &r);
		// Knu_e[i]=r.val;

		// gsl_sf_bessel_Knu_scaled_asymp_unif_e(2.,x[i], &r);
		// Knu_unife[i]=r.val;

		// irregular modified cylindrical Bessel function of order n, K_n(x)
		Kn[i]=gsl_sf_bessel_Kn(2,x[i]);
	}

	// write file
	printf("Saving file %s\n", filename);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (i=0; i<N; i++) {
        fprintf(f, "%f %f %f %f \n", x[i], K0[i], K1[i], Kn[i]);
    }

    fclose(f);
    free(x);
    free(K0);
    free(K1);
    // free(Kn_small);
    // free(Knu_e);
    // free(Knu_unife);
    free(Kn);

	return 0;
}