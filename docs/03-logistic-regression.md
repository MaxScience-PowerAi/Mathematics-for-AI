# Régression logistique

La régression logistique binaire modélise la probabilité conditionnelle d'une classe $Y=1$ sachant un vecteur de caractéristiques $x$. On pose

$$p(x)=\sigma(x^Tw),\qquad \sigma(z)=\frac{1}{1+e^{-z}}.$$

Le modèle est linéaire dans les log-odds : $\log(p/(1-p))=x^Tw$. Une colonne de uns permet d'inclure un biais.

## Vraisemblance et gradient

Pour des observations binaires conditionnellement indépendantes, maximiser la vraisemblance de Bernoulli revient à minimiser l'entropie croisée binaire. Avec une pénalisation quadratique $\lambda\ge0$ :

$$J(w)=\frac1n\sum_{i=1}^n\log(1+\exp(-(2y_i-1)x_i^Tw))+\frac{\lambda}{2}\|w\|_2^2.$$

En notant $p=\sigma(Xw)$, on obtient

$$\nabla J(w)=\frac1n X^T(p-y)+\lambda w.$$

La forme softplus, calculée par numpy.logaddexp, évite de prendre le logarithme d'une probabilité arrondie à zéro. Dans le code, le gradient utilise une forme signée équivalente à $p-y$.

La Hessienne est

$$H(w)=\frac1nX^T\operatorname{diag}(p_i(1-p_i))X+\lambda I.$$

Comme $0\le p_i(1-p_i)\le1/4$, la fonction est convexe et son gradient est Lipschitz avec la borne $L=\|X\|_2^2/(4n)+\lambda$. Pour $L>0$, le code utilise le pas $1/L$. Le cas $L=0$ correspond ici à une fonction constante. Tous les coefficients sont régularisés, biais compris.

Sans régularisation, des données parfaitement séparables peuvent conduire à une vraisemblance dont le supremum n'est atteint par aucun vecteur fini. Une pénalisation strictement positive assure un minimiseur unique.

## Expérience reproductible

Depuis la racine du dépôt :

```python
import numpy as np
from math_for_ai.logistic import fit, sigmoid

X = np.column_stack([np.ones(6), [-3, -2, -1, 1, 2, 3]])
y = np.array([0, 0, 0, 1, 1, 1])
w, history = fit(X, y, steps=200, l2=.01)
print(w)
print(history[0], history[-1])
print(sigmoid(X @ w))
```

Résultat local arrondi : poids $(0, 2{,}627264)$, objectif initial $0{,}693147$, objectif final $0{,}059636$. Les six exemples d'entraînement sont classés correctement avec un seuil de 0,5. Cela ne mesure pas la généralisation : les données sont synthétiques, séparables et ne comportent pas de jeu de test indépendant.

## À expérimenter

1. Comparer la valeur du gradient analytique à une différence finie centrée.
2. Modifier $\lambda$ et observer les poids et les probabilités.
3. Générer des données non séparables, puis créer un jeu de test indépendant.
4. Comparer la pénalisation du biais à une version qui ne le pénalise pas.

L'implémentation privilégie la lisibilité plutôt que les performances. Elle n'effectue ni standardisation automatique, ni sélection d'hyperparamètres, ni évaluation indépendante.

[Code](../math_for_ai/logistic.py) · [Tests](../tests/test_logistic.py)

Ressource : [Stanford CS229, lecture notes](https://cs229.stanford.edu/main_notes.pdf), section sur la régression logistique.
