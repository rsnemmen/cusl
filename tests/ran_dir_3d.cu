/*
  This program uses CURAND on the device. Each thread will 
  generate a random number from a uniform distribution between 0 and 1. 
  It will print the numbers but not return them to the host.
*/
#include <stdio.h> 
#include <stdlib.h> 
#include <curand_kernel.h>
#include "sphere.cuh" // imports cu_ran_dir_3d

#define TPB 64  
#define SEED 42



__global__ void generate(curandState *state, int n){ 
	int id = blockIdx.x*blockDim.x + threadIdx.x; 
	double x, y, z, norm;
	curandState r;

	if (id>=n) return;

	r=state[id];
	curand_init(SEED, id, 0, &r);	

	for (int i=0; i<n; i++) {
		cu_ran_dir_3d_double(&r, &x, &y, &z);
		norm=sqrt(x*x+y*y+z*z);

		printf("i=%d x=%f y=%f z=%f norm=%f\n", i, x, y, z, norm);
	}

	state[id]=r;
}



int main(int argc, char *argv[]){
	int n, onethread;
	/* A note about RNGs:

	   • use curandState if you want to use the default XORWOW generator
	   • use curandStateMRG32k3a if you want to use the Mersenne Twister
	*/
	curandState *d_states ; // XORWOW

    // handle command-line argument
    if ( argc != 3 ) {
        printf( "usage: %s <number of randoms desired> <1 thread or multiple? (1 or 0)> \n", argv[0] );
        exit(0);
    }  
    // n = number of random numbers (and threads)
    sscanf(argv[1], "%d", &n); 
    sscanf(argv[2], "%d", &onethread); 

	/* Allocate space for prng states on device */
    cudaMalloc((void **)&d_states , n*sizeof(curandState));

    printf("Generating RNs on the GPU, ");
    if (onethread) {
		printf("one thread\n");
		generate<<<1,1>>>(d_states, n); 	
	}
	else {
		printf("multiple threads\n");
		generate<<<(n+TPB-1)/TPB, TPB>>>(d_states, n); 
	}
	
	/* Cleanup */ 
	cudaFree(d_states); 

	return 0; 
}