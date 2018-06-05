/*
  This program uses CURAND on the device. Each thread will 
  generate a random number from a uniform distribution between 0 and 1. 
  It will print the numbers but not return them to the host.
*/
#include <stdio.h> 
#include <stdlib.h> 

#define TPB 64  

#define N 100
#define x0 0.5
#define x1 3.
#define filename "bessel-cuda.dat"



__global__ void bessel(double *d_x, double *d_y, int n){ 
	int id = blockIdx.x*blockDim.x + threadIdx.x; 

	if (id>=n) return;

	d_y[id]=yn(2,d_x[id]);
}



int main(int argc, char *argv[]){
	int i;
	double *x = (double *)malloc(N*sizeof(double));
	double *y = (double *)malloc(N*sizeof(double));	

	for (i=0; i<N; i++) {
		// generate array of x-values
		x[i]=x0+i*(x1-x0)/(N-1);
	}	

	// send to GPU
	double *d_x, *d_y;
	cudaMalloc(&d_x, N*sizeof(double));	
	cudaMalloc(&d_y, N*sizeof(double));	
	cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);

	// kernel launch
    printf("Computing Bessel function on the GPU\n");
	bessel<<<(N+TPB-1)/TPB, TPB>>>(d_x, d_y, N); 
	cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);	
	
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

    // clean up
    fclose(f);
    free(x);
    free(y);
	cudaFree(d_x);
	cudaFree(d_y); 

	return 0; 
}