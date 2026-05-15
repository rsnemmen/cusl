# Validation

CUSL validation is currently based on small executable comparisons in `tests/`.
The tests are useful for development, but they are not yet a formal automated
test suite.

## Bessel Comparison

```sh
cd tests
sh bessel.sh
```

The script:

- builds a GSL reference program with `gcc`
- builds the CUSL CUDA implementation with `nvcc`
- builds a CUDA-standard comparison program with `nvcc`
- runs all three programs
- generates comparison output and a plot with Python

The CUSL program evaluates:

- `cu_sf_bessel_K0_scaled_e`
- `cu_sf_bessel_K1_scaled_e`
- `cu_sf_bessel_Kn` for integer orders 2 through 5

The comparison range in the current CUDA test is `x = 0.1` to `x = 20.0` with
100 samples.

## Random Direction Comparison

```sh
cd tests
sh ran_dir_3d.sh
```

The script builds and runs a GSL reference program and a CUDA/cuRAND program for
`cu_ran_dir_3d_double`.

The current comparison prints sample vectors and norms. It does not yet perform
a full distribution test. Because GSL and cuRAND do not use the same generator
state, exact sample-by-sample equality is not expected.

## Validation Gaps

- Add distribution-level tests for `cu_ran_dir_3d_double`.
- Add distribution-level tests for `cu_ran_chisq_double`.
- Add dedicated tests for `cu_ran_gamma_double`, `cu_sf_fact_e`,
  `cu_sf_lnfact_e`, and `cu_sf_psi_int_e`.
- Add boundary-case tests for small arguments, large arguments, domain errors,
  and overflow behavior.
- Decide whether future validation should remain shell-script based or move to
  a lightweight build/test harness.
