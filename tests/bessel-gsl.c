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
#define x1 20.
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
	double *K2 = (double *)malloc(N*sizeof(double));
	double *K3 = (double *)malloc(N*sizeof(double));
	double *K4 = (double *)malloc(N*sizeof(double));
	double *K5 = (double *)malloc(N*sizeof(double));

	for (i=0; i<N; i++) {
		// generate array of x-values
		x[i]=x0+i*(x1-x0)/(N-1);

		// scaled irregular modified cylindrical Bessel function of zeroth order \exp(x) K_0(x)
		gsl_sf_bessel_K0_scaled_e(x[i], &r);
		K0[i]=r.val;

		// irregular modified cylindrical Bessel function of first order, K_1(x), for x > 0.
		gsl_sf_bessel_K1_scaled_e(x[i], &r);
		K1[i]=r.val;

		// bessel_Kn_scaled_small_x(2, x[i], &r);
		// Kn_small[i]=r.val;

		// gsl_sf_bessel_Knu_scaled_asympx_e(2.,x[i], &r);
		// Knu_e[i]=r.val;

		// gsl_sf_bessel_Knu_scaled_asymp_unif_e(2.,x[i], &r);
		// Knu_unife[i]=r.val;

		// irregular modified cylindrical Bessel function of order n, K_n(x)
		K2[i]=gsl_sf_bessel_Kn(2,x[i]);
		K3[i]=gsl_sf_bessel_Kn(3,x[i]);
		K4[i]=gsl_sf_bessel_Kn(4,x[i]);
		K5[i]=gsl_sf_bessel_Kn(5,x[i]);
	}

	// write file
	printf("Saving file %s\n", filename);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(f, "# x K0_scaled K1_scaled K2 K3 K4 K5\n");
    for (i=0; i<N; i++) {
        fprintf(f, "%f %f %f %f %f %f %f\n", x[i], K0[i], K1[i], K2[i], K3[i], K4[i], K5[i]);
    }

    fclose(f);
    free(x);
    free(K0);
    free(K1);
    // free(Kn_small);
    // free(Knu_e);
    // free(Knu_unife);
    free(K2);
    free(K3);
    free(K4);
    free(K5);

	return 0;
}
