# Repository Guidelines

## Project Structure & Module Organization

CUSL is a small CUDA port of selected GNU Scientific Library routines. Core code lives in `src/` as reusable headers:

- `src/*.cuh`: CUDA implementations for Bessel, Gamma, Psi, polynomial, Chebyshev, chi-square, and sphere/random-direction routines.
- `src/cusl_math.h` and `src/gsl_machine.h`: shared constants and compatibility definitions.
- `tests/`: standalone C, CUDA, shell, Python, and reference data files for comparing CUSL with GSL and CUDA standard functions.

There is no central build system; compile tests from `tests/`.

## Build, Test, and Development Commands

Run from `tests/` unless noted.

- `sh bessel.sh`: compiles and runs the GSL, CUSL, and CUDA-standard Bessel tests, then generates comparison data/plots.
- `gcc bessel-gsl.c -o bessel-gsl -lgsl -lm -lgslcblas`: builds the GSL reference executable.
- `nvcc bessel-cusl.cu -o bessel-cusl -I../src`: builds a CUDA test against headers in `src/`.
- `nvcc ran_dir_3d.cu -o ran_dir_3d -I../src -lcurand`: builds the random-direction CUDA test if you need to run it directly.

Required tools: `nvcc`, CUDA/cuRAND, `gcc`, GSL development libraries, and Python.

## Coding Style & Naming Conventions

Follow the existing C/CUDA style: K&R-like braces, short helper functions, `#define` constants for numeric parameters, and direct CUDA kernel launches in tests. Use tabs or 4-space indentation consistently within the file being edited. Keep public CUDA/GSL-compatible functions prefixed with `cu_` or `cu_sf_`, matching names such as `cu_sf_bessel_Kn` and `cu_ran_dir_3d_double`.

Header additions should stay self-contained.

## Testing Guidelines

Tests are executable comparisons rather than a formal framework. Prefer adding a GSL or known-reference counterpart in `tests/` for each new numerical routine. Name files after the routine, for example `bessel-cusl.cu`, `bessel-gsl.c`, or `<routine>.sh`.

When changing numerical code, run the relevant shell script or compile both the CUDA and reference programs manually. Inspect generated `.dat` files or plots for agreement, especially around small-argument and boundary cases.

## Commit & Pull Request Guidelines

The current history uses short, imperative or descriptive commit subjects such as `removed temp file` and `Debye functions`. Keep commits focused on one routine, test, or cleanup. In pull requests, include the routine changed, the commands run, and any numerical reference used. Attach plots or data snippets when a change affects computed values.

## Agent-Specific Instructions

Do not add generated binaries, plots, `.dat` outputs, or system files such as `.DS_Store` unless explicitly requested. Keep documentation updates concise and aligned with this repository’s lightweight workflow.
