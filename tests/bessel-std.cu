/*
  Generate values from standard Bessel functions that come with the
  CUDA Math Library.

  cf. https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE
*/
#include <stdio.h>
#include <stdlib.h>

#define TPB 64

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(1); \
	} \
} while (0)

#define N 100
#define x0 0.1
#define x1 5.
#define filename "bessel-cuda-std.dat"



__global__ void bessel(double *x, double *y_n, double *i0, double *i1, int n){
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id>=n) return;

	y_n[id] =yn(2,x[id]);
	i0[id] =cyl_bessel_i0(x[id]);
	i1[id] =cyl_bessel_i1(x[id]);
}



int main(int argc, char *argv[]){
	int i;
	double *x, *y_n, *i0, *i1;

	// Allocate Unified Memory - accessible from CPU or GPU
	CUDA_CHECK(cudaMallocManaged(&x, N*sizeof(double)));
	CUDA_CHECK(cudaMallocManaged(&y_n, N*sizeof(double)));
	CUDA_CHECK(cudaMallocManaged(&i0, N*sizeof(double)));
	CUDA_CHECK(cudaMallocManaged(&i1, N*sizeof(double)));

	for (i=0; i<N; i++) {
		// generate array of x-values
		x[i]=x0+i*(x1-x0)/(N-1);
	}

	// kernel launch
    printf("Computing Bessel function on the GPU\n");
	bessel<<<(N+TPB-1)/TPB, TPB>>>(x, y_n, i0, i1, N);
	CUDA_CHECK(cudaGetLastError());

	// Wait for GPU to finish before accessing on host
	CUDA_CHECK(cudaDeviceSynchronize());

	// write file
	printf("Saving file %s\n", filename);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (i=0; i<N; i++) {
        fprintf(f, "%f %f %f %f \n", x[i], y_n[i], i0[i], i1[i]);
    }

    // clean up
    fclose(f);
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y_n));
	CUDA_CHECK(cudaFree(i0));
	CUDA_CHECK(cudaFree(i1));

	return 0;
}
