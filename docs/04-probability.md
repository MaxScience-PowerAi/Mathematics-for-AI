# Probabilités et simulation Monte-Carlo

## Loi de Bernoulli

Une variable $X$ de Bernoulli de paramètre $p\in[0,1]$ vaut 1 avec probabilité $p$ et 0 sinon. Comme $X^2=X$,

$$E[X]=p,\qquad \operatorname{Var}(X)=E[X^2]-E[X]^2=p(1-p).$$

Pour $n$ observations indépendantes de même loi, la moyenne $\hat p=n^{-1}\sum_{i=1}^n X_i$ est un estimateur sans biais de $p$ :

$$E[\hat p]=p,\qquad \operatorname{Var}(\hat p)=\frac{p(1-p)}{n}.$$

L'erreur-type théorique est donc $\sqrt{p(1-p)/n}$. En pratique, si $p$ est inconnu, on peut estimer l'erreur-type par $s/\sqrt n$, où $s^2$ est la variance empirique corrigée. Cette quantité n'est pas une borne garantie de l'erreur absolue.

## Expérience

Depuis la racine du dépôt, avec NumPy :

```python
from math_for_ai.probability import bernoulli_trial_mean, theoretical_standard_error

for n in (100, 1000, 10000):
    mean, se = bernoulli_trial_mean(.3, n, seed=42)
    print(n, mean, se, theoretical_standard_error(.3, n))
```

Résultats obtenus lors de la préparation assistée du module (NumPy 2.3.5, graine 42) :

| n | Moyenne | Erreur-type estimée | Erreur-type théorique |
|---:|---:|---:|---:|
| 100 | 0,2700 | 0,04462 | 0,04583 |
| 1 000 | 0,3040 | 0,01455 | 0,01449 |
| 10 000 | 0,2942 | 0,00456 | 0,00458 |

La loi des grands nombres assure la convergence de la moyenne vers $p$. Elle ne garantit pas une diminution monotone de l'erreur absolue pour chaque réalisation. Ici, l'erreur absolue passe de 0,004 à 0,0058 entre les deux dernières tailles, alors que l'erreur-type théorique diminue.

Multiplier $n$ par quatre divise l'erreur-type par deux. Le même principe intervient dans l'estimation de quantités par simulation Monte-Carlo, sous les hypothèses d'indépendance et de variance finie.

## Lien avec l'apprentissage automatique

La vraisemblance de Bernoulli est au cœur de la régression logistique binaire. Dans ce modèle, la probabilité de succès dépend des caractéristiques. Voir [la régression logistique](03-logistic-regression.md).

## Exercices à réaliser

- Calculer l'erreur-type de la moyenne pour $p=0,5$ et $n=400$ (réponse : 0,025).
- Répéter l'expérience avec plusieurs graines et comparer l'écart-type des moyennes à la formule théorique.
- Expliquer pourquoi une variance empirique nulle pour un petit échantillon n'implique pas que $p$ soit connu exactement.

Les exemples utilisent un générateur pseudo-aléatoire et des données simulées, sans jeu de données réel. La graine rend l'expérience reproductible dans un environnement compatible ; les résultats précis peuvent dépendre de la version du générateur.

[Implémentation](../math_for_ai/probability.py) · [Tests](../tests/test_probability.py)
