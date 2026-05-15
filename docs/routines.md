# Routines

The table below lists the currently visible CUSL routines and their closest GSL
counterparts.

| GSL routine or concept | CUSL routine | Header | Status | Validation |
| --- | --- | --- | --- | --- |
| `gsl_ran_dir_3d` | `cu_ran_dir_3d_double` | `sphere.cuh` | Ported | `tests/ran_dir_3d.sh` compares sample output and vector norms |
| `gsl_sf_bessel_K0_scaled_e` | `cu_sf_bessel_K0_scaled_e` | `bessel.cuh` | Ported | `tests/bessel.sh` compares against GSL |
| `gsl_sf_bessel_K1_scaled_e` | `cu_sf_bessel_K1_scaled_e` | `bessel.cuh` | Ported | `tests/bessel.sh` compares against GSL |
| `gsl_sf_bessel_Kn` | `cu_sf_bessel_Kn` | `bessel.cuh` | Ported for integer order | `tests/bessel.sh` compares orders 2 through 5 |
| `gsl_ran_chisq` | `cu_ran_chisq_double` | `chisq.cuh` | Ported | Distribution validation still needed |
| `gsl_ran_gamma` | `cu_ran_gamma_double` | `gamma.cuh` | Ported as a dependency for random distributions | Dedicated validation still needed |
| `gsl_sf_lnfact_e` | `cu_sf_lnfact_e` | `gamma.cuh` | Ported | Dedicated validation still needed |
| `gsl_sf_fact_e` | `cu_sf_fact_e` | `gamma.cuh` | Ported | Dedicated validation still needed |
| `gsl_sf_psi_int_e` | `cu_sf_psi_int_e` | `psi.cuh` | Ported | Dedicated validation still needed |
| GSL polynomial helper | `cu_poly_eval` | `poly.cuh` | Helper routine | Covered indirectly by Bessel validation |
| GSL Chebyshev helper | `cheb_eval_e` | `cheb_eval.cuh` | Helper routine | Covered indirectly by Bessel validation |

## Public API Shape

Most routines are CUDA `__device__` functions. They are intended to be called
from kernels or other device functions.

CUSL names generally use one of these prefixes:

- `cu_sf_` for special functions
- `cu_ran_` for random distributions
- `cu_` for lower-level helpers

Unlike GSL, the current CUSL routines do not expose `gsl_sf_result` output
objects or integer status codes.
