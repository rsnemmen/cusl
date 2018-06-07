CUSL: A CUDA port of the GNU Scientific Library
=================================================

Port of some routines of the GNU Scientific Library (GSL) to CUDA.

# Dependencies

cuRAND

# Available functions

- gsl_ran_dir_3d => sphere.cuh/cu_ran_dir_3d

cu_sf_bessel_K0_scaled_e
cu_sf_bessel_K1_scaled_e
gsl_sf_bessel_Kn

- gsl_ran_chisq

cheb_eval_e

gsl_sf_lnfact_e
gsl_sf_fact_e: factorial
gsl_ran_gamma_double: Gamma distribution

cu_sf_psi_int_e: Digamma (Psi) function

cu_poly_eval: polynomial evaluation


# Work needed: TODO

- [x] cu_ran_dir_3d test
- [ ] further tests of distribution of vectors from cu_ran_dir_3d, comparison with GSL. Since we do not have the same RNG available in the GPU and GSL, we cannot test random vectors for the same random numbers 
- [ ] test cu_sf_bessel_Kn
- [ ] cu_ran_chisq: check if distribution is correct and matches GSL
- [ ] error checking like GSL: I removed all calls to `gsl_sf_result` to simplify things, so there is very little automatic error checking in the CUDA routines available here