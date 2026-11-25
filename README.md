# MATHEMATICS / FOR AI

**From mathematical foundations to intelligent systems.**

A bilingual-oriented learning repository for the numerical foundations of AI. Prepared as an AI-assisted educational starter for the learning journey of Linze Lowe Christ Maxime (Christ Lowe). It contains worked examples, not original research results or evidence of independent authorship. Review and extend the material through your own experiments before presenting it as personal work.

[Personal platform / EN](https://christ-lowe-portfolio.vercel.app/en) · [Plateforme / FR](https://christ-lowe-portfolio.vercel.app/fr)

## 01 / Available modules

| Module | Theory | Implementation |
|---|---|---|
| Linear algebra | [Notes and exercises / FR](docs/01-linear-algebra.md) | [Python](math_for_ai/linear_algebra.py) |
| Optimization | [Notes and exercises / FR](docs/02-optimization.md) | [Python](math_for_ai/optimization.py) |
| Computational experiments | [Jupyter notebook / FR–EN](notebooks/01-foundations.ipynb) | Synthetic deterministic examples |

The initial release covers cosine similarity, orthogonal projections, least squares, truncated SVD, convergence of gradient descent on a scalar quadratic, and full-batch linear regression. All calculations run locally on CPU; no API, GPU or paid service is required.

## 02 / Reproduce

From the repository root with Python 3.11 or later (initial validation used Python 3.13):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m examples.run_demo
```

For the notebook:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

Open `notebooks/01-foundations.ipynb` and run all cells. On Windows, activate the environment with `.venv\Scripts\Activate.ps1` instead. All data are synthetic. See [validation](docs/VALIDATION.md) for the actual execution environment and limits.

## 03 / Scientific roadmap

The following modules are planned, not yet implemented:

- Probability, random variables, expectations and Monte Carlo.
- Statistical inference and uncertainty.
- Multivariable calculus, Jacobians and Hessians.
- Convex optimization, regularization and numerical conditioning.
- Information theory, entropy and cross-entropy.
- Eigenvalues, PCA and dimensionality reduction.

## 04 / References

- Boyd & Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).
- Deisenroth, Faisal & Ong, [Mathematics for Machine Learning](https://mml-book.github.io/).
- [NumPy linear algebra documentation](https://numpy.org/doc/stable/reference/routines.linalg.html).

These are further-reading references; the notes and code are introductory worked examples. Dependencies are version-bounded rather than fully locked. CI is supplied but must be verified after the initial GitHub push. A repository license has not yet been selected by the owner.

**En français.** Ce dépôt est une base pédagogique consacrée aux mathématiques de l'IA. Il relie les formules à des implémentations NumPy, à des tests et à des expériences reproductibles. Les ressources initiales sont préparées avec assistance IA et restent à approfondir personnellement. Aucun résultat de recherche inédit n'est revendiqué.
