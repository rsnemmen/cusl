# Porting Notes

CUSL ports selected GSL algorithms into CUDA device code while keeping the
implementation compact and easy to inspect.

## Conventions

- Keep headers self-contained where practical.
- Use CUDA `__device__` functions for routines intended to run inside kernels.
- Preserve GSL-compatible naming where useful, replacing `gsl_` with `cu_`.
- Prefer direct numerical ports over new abstractions unless shared helpers make
  the code clearer.
- Add a GSL or known-reference comparison program when adding a new numerical
  routine.

## Error Handling

GSL commonly returns status codes and writes values plus error estimates through
`gsl_sf_result`. CUSL currently simplifies this: routines generally return a
single `double`, and some invalid inputs return `CUSL_NAN`.

This is the largest API difference from GSL. Future work should decide whether
to:

- keep the current compact `double`-returning API
- add status/result variants for routines that need stronger error reporting
- use both forms, with simple variants wrapping result/status variants

## Documentation Policy

The public documentation should describe stable, user-facing behavior:

- routine coverage
- include requirements
- valid input domains
- known limitations
- validation commands and reference sources

Avoid documenting internal coefficient tables or helper implementation details
unless they are needed to understand numerical accuracy or porting provenance.

## Roadmap

- Expand the routine coverage table as new GSL routines are ported.
- Add validation notes next to each public routine.
- Improve error handling policy before exposing many more special functions.
- Consider Doxygen later if the headers gain structured doc comments and a
  larger public API surface.
