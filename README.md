CUSL: A CUDA port of the GNU Scientific Library
=================================================

CUSL is an experimental CUDA port of selected GNU Scientific Library (GSL)
routines. It is a small, header-oriented project focused on CUDA device code,
not a complete GSL replacement.

Full documentation is available at:

- https://rsnemmen.github.io/cusl/

## Quick start

Run from `tests/`:

```sh
sh bessel.sh
sh ran_dir_3d.sh
```

These scripts compile and run small comparison programs against GSL and CUDA
reference implementations.

## Documentation

The documentation covers:

- available routines and GSL-to-CUSL mapping
- dependencies and build commands
- validation scripts and known test gaps
- porting notes for future routines

To preview the documentation locally:

```sh
python -m pip install -r requirements-docs.txt
mkdocs serve
```

## Status

- `src/` contains CUDA headers for selected Bessel, Gamma, Psi, polynomial,
  Chebyshev, chi-square, and random-direction routines.
- `tests/` contains standalone validation programs and scripts.
- GSL-style result/status error handling is not fully implemented.
