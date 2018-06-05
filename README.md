CUSL: A CUDA port of the GNU Scientific Library
=================================================

Port of some routines of the GNU Scientific Library (GSL) to CUDA.

# Dependencies

cuRAND

# Available functions

- [x] gsl_ran_dir_3d => sphere.cuh/cu_ran_dir_3d
- [ ] gsl_sf_bessel_Kn
- [x] gsl_ran_chisq

# Work needed: TODO

- [x] cu_ran_dir_3d (partially done)
- [ ] test cu_sf_bessel_Kn
- [ ] test cu_ran_chisq