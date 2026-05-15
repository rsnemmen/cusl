# CUSL

CUSL is an experimental CUDA port of selected routines from the GNU Scientific
Library (GSL). It is intended for CUDA device code that needs a small subset of
GSL-style special functions and random distributions without calling back to the
CPU.

The project is not a complete GSL replacement. The current code is a compact,
header-oriented collection of ports in `src/`, with validation programs in
`tests/` that compare CUSL output against GSL or CUDA standard functions.

## Current Status

- Header-only CUDA/C++ implementation files live in `src/*.cuh`.
- Existing validation is executable comparison code, not a formal test suite.
- GSL-style status objects and error propagation are not implemented. Most
  routines return a `double`; domain and overflow cases may return `CUSL_NAN`
  and some device routines print diagnostic messages.
- Distribution-level statistical validation is still incomplete for some random
  routines.

## Repository Layout

```text
src/     CUDA headers implementing selected GSL routines and helpers
tests/   GSL, CUDA, shell, Python, and reference-data comparison programs
```

## Documentation Map

- [Getting Started](getting-started.md): dependencies, build commands, and a
  minimal CUDA usage example.
- [Routines](routines.md): current GSL-to-CUSL coverage table.
- [Validation](validation.md): available comparison scripts and what they check.
- [Porting Notes](porting-notes.md): conventions and known limitations for
  future ports.
