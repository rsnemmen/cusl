#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>

#define N 100
#define x0 0.5
#define x1 3.
#define filename "bessel.dat"

int main(int argc, char const *argv[])
{
	int i;
	double *x = (double *)malloc(N*sizeof(double));
	double *y = (double *)malloc(N*sizeof(double));	

	for (i=0; i<N; i++) {
		// generate array of x-values
		x[i]=x0+i*(x1-x0)/(N-1);

		// compute Bessel
		y[i]=gsl_sf_bessel_Kn(2,x[i]);
	}

	// write file
	printf("Saving file %s\n", filename);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (i=0; i<N; i++) {
        fprintf(f, "%f %f \n", x[i], y[i]);
    }

    fclose(f);
    free(x);
    free(y);

	return 0;
}