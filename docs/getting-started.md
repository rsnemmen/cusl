# Getting Started

CUSL is currently used by including the relevant CUDA headers directly from
`src/`. There is no central build system.

## Dependencies

For compiling and validating the current code, install:

- CUDA Toolkit with `nvcc`
- cuRAND
- `gcc`
- GSL development libraries, including `gsl` and `gslcblas`
- Python for plotting scripts

## Build A Validation Program

Run commands from `tests/` unless noted otherwise.

```sh
cd tests
nvcc bessel-cusl.cu -o bessel-cusl -I../src
```

The random-direction validation program also links cuRAND:

```sh
cd tests
nvcc ran_dir_3d.cu -o ran_dir_3d -I../src -lcurand
```

## Run Existing Comparisons

```sh
cd tests
sh bessel.sh
sh ran_dir_3d.sh
```

These commands generate local binaries, data files, and plots. Do not commit
generated binaries, generated `.dat` outputs, or plots unless they were
explicitly requested.

## Minimal CUDA Usage Example

```cuda
#include "cusl_math.h"
#include "cheb_eval.cuh"
#include "poly.cuh"
#include "psi.cuh"
#include "gamma.cuh"
#include "bessel.cuh"

__global__ void kernel(double *x, double *out, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    out[id] = cu_sf_bessel_Kn(2, x[id]);
}
```

The Bessel implementation currently depends on helper headers such as
`cheb_eval.cuh`, `poly.cuh`, `psi.cuh`, and `gamma.cuh`. Include those helpers
before `bessel.cuh`, following the existing validation program.
