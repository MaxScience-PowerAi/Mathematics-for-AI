# Mathematics for AI

Notes and Python experiments on the mathematics behind machine learning and artificial intelligence, with a focus on understanding the methods rather than simply using libraries.

## Topics

| Topic | Notes and exercises | Python implementation |
| --- | --- | --- |
| Linear algebra | [Read](docs/01-linear-algebra.md) | [Code](math_for_ai/linear_algebra.py) |
| Optimization | [Read](docs/02-optimization.md) | [Code](math_for_ai/optimization.py) |
| Logistic regression | [Read](docs/03-logistic-regression.md) | [Code](math_for_ai/logistic.py) |
| Probability and simulation | [Read](docs/04-probability.md) | [Code](math_for_ai/probability.py) |

The examples cover cosine similarity, orthogonal projection, least squares, truncated SVD, gradient descent, linear regression and binary logistic regression and Bernoulli simulation. The [Jupyter notebook](notebooks/01-foundations.ipynb) brings these concepts together using synthetic data.

## Getting started

Python 3.11 or later is required. No GPU or external API is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m examples.run_demo
```

To explore the notebook:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

Open `notebooks/01-foundations.ipynb` and run the cells. On Windows, use `.venv\Scripts\Activate.ps1` to activate the environment instead.

See the [validation notes](docs/VALIDATION.md) for the testing environment and limitations.

## Further topics

Statistical inference, multivariable calculus, convex optimization, information theory and principal component analysis are areas for future study.

## References

- Deisenroth, Faisal & Ong — [Mathematics for Machine Learning](https://mml-book.github.io/)
- Boyd & Vandenberghe — [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
- [NumPy linear algebra documentation](https://numpy.org/doc/stable/reference/routines.linalg.html)

## About

Maintained as part of the learning journey of Linze Lowe Christ Maxime. The notes and code have been prepared with AI assistance; the repository is educational, not a claim of original research. A project license has not yet been selected.

[Portfolio](https://christ-lowe-portfolio.vercel.app/en) · [Version française](https://christ-lowe-portfolio.vercel.app/fr)

Les notes mathématiques et les exercices sont disponibles en français dans le dossier [docs](docs).
