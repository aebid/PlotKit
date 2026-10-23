# PlotKit — instructions for Copilot code review

A standalone CMS plotting toolkit (matplotlib/mplhep, with a ROOT/cmsstyle backend) used by
[FLAF](https://github.com/cms-flaf/FLAF) as a submodule and usable on its own through its CLI.

**Read `FLAF/.github/copilot-instructions.md` first** for the shared rules on what a useful review
comment looks like here and what not to flag. The rule that documentation ships in the same PR
applies here too, and is restated below.

## What makes this repository different

Unlike the rest of the ecosystem, PlotKit is an ordinary installable Python package with **real
unit tests that run anywhere** — no CVMFS, no grid proxy, no CMSSW. So the usual excuse does not
apply here: a change to `spec.py`, `config.py`, `histogram.py`, `rootcompat.py` or a plotter
should come with a test, and asking for one is a legitimate review comment.

It is also consumed by FLAF *transparently*: analyses call it through FLAF, not directly. A
change to a public name or to the shape of a spec is therefore an API break that surfaces in three
analyses, not a local refactor.

## Invariants

### Backend parity

`plotters/` renders through matplotlib/mplhep and through ROOT/cmsstyle. A feature added to one
backend and not the other produces plots that differ depending on which is selected — silently,
because both succeed. Check that a new option is either implemented in both or explicitly
rejected by the one that does not support it.

### Specs and configuration are a contract

`spec.py` and `config.py` define what a caller may pass. Renaming a key, changing a default, or
tightening validation changes behaviour for every existing configuration. Backwards-incompatible
changes need to be called out; silent default changes are the ones that get noticed months later
in a plot that looks subtly different.

### ROOT is optional

`rootcompat.py` exists so the package works without ROOT installed. An import of ROOT at module
scope, or a code path that assumes it is present, breaks the pure-matplotlib installation. Check
that new ROOT usage stays behind the compatibility layer.

### Plot correctness is not testable by eye

Look for the things a screenshot would not reveal: overflow and underflow silently dropped, bin
edges assumed uniform, error propagation on a ratio, normalisation applied twice, a stack ordered
by dictionary iteration order.

## Documentation must ship with the change

A PR must update the documentation **in the same PR** whenever it changes anything a user can
observe: a CLI flag, a spec or config key, a default, a plotter option, or the meaning of an
existing one.

This repository documents itself in `README.md` and `examples/`; keep both truthful — an example
that no longer runs is worse than no example. Where the change affects how FLAF drives plotting,
the framework documentation in `FLAF/docs/` needs a companion PR against `cms-flaf/FLAF`; flag
its absence.

## Do not flag

- The matplotlib/ROOT dual-backend structure itself.
- `from __future__` style questions or minor typing gaps in plotting code.
- Formatting — the `formatting-check` workflow settles it.

## Repository facts

Verified 2026-10-23; re-check before relying on any of it.

| | |
|---|---|
| Layout | `Plotter.py`, `plotters/` (`base.py`, `stacked.py`), `spec.py`, `config.py`, `histogram.py`, `backends.py`, `rootcompat.py`, `cli.py`, `examples/`, `tests/` |
| Packaging | `pyproject.toml` (setuptools); depends on numpy, matplotlib, mplhep, pyyaml, uproot. ROOT is optional |
| Tests | `tests/` — `test_cli.py`, `test_config.py`, `test_rootcompat.py`, `test_smoke.py`, run with pytest |
| Consumed as | a submodule of FLAF at `FLAF/PlotKit`; the fork remote is `kandrosov/FLAF-PlotKit` |
| Workflows | `formatting-check`, `repo-sanity-checks`, `trigger-flaf-integration`, `unit-tests` (pytest on PRs and pushes to main: `unit-tests` with flaf_env's plotting pins and no ROOT, so the cmsstyle tests skip; `cmsstyle-tests` with conda-forge ROOT and cmsstyle, where `PLOTKIT_REQUIRE_ROOT` makes them fail instead of skip) |
