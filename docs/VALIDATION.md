# Validation — initial release

Local environment of the prepared starter: Python 3.13.5, NumPy 2.3.5, Matplotlib 3.10.8.

- PASS: 22 unit tests (`python -m unittest discover -s tests -v`).
- PASS: all four notebook code cells executed using nbclient; notebook schema validated.
- PASS: deterministic linear regression compared against NumPy least squares.
- Notebook source is published with cleared outputs for a clean starting point.
- GitHub Actions: configured; remote status must be checked after publication.
- Python 3.11/3.12: CI targets, not verified locally.
- Dependencies: bounded ranges, not an exact environment lock.
- Scope: AI-assisted educational examples with synthetic data, not scientific peer review or a performance benchmark.
