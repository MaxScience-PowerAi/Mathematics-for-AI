# 02 — Optimisation et descente de gradient

## 1. Quadratique scalaire

Soit $f(x)=\kappa x^2/2$, avec $\kappa>0$. Le minimiseur unique est $0$ et $f'(x)=\kappa x$. Pour un pas constant $\eta>0$ :

$$x_{n+1}=x_n-\eta f'(x_n)=(1-\eta\kappa)x_n.$$

Par récurrence, $x_n=(1-\eta\kappa)^n x_0$. Si $x_0\ne0$, la convergence vers zéro équivaut à $|1-\eta\kappa|<1$, soit $0<\eta\kappa<2$. Si $x_0=0$, la suite reste nulle dans le modèle exact, quel que soit le pas.

| Produit $\eta\kappa$ | Comportement pour $x_0\ne0$ |
|---|---|
| Entre 0 et 1 | Convergence directe |
| 1 | Minimum atteint en un pas |
| Entre 1 et 2 | Convergence oscillante |
| 2 | Oscillation d'amplitude constante |
| Supérieur à 2 | Divergence |

Ce résultat exact est propre à la quadratique positive à une dimension. L'implémentation utilise des nombres flottants et signale les débordements numériques.

## 2. Régression linéaire par gradient

Avec $X\in\mathbb R^{n\times d}$, $y\in\mathbb R^n$, on minimise

$$J(w)=\frac1{2n}\|Xw-y\|_2^2,$$

$$\nabla J(w)=\frac1n X^T(Xw-y),\qquad H=\frac1n X^TX.$$

La mise à jour de gradient complet est $w_{t+1}=w_t-\eta\nabla J(w_t)$. Si $X$ est de rang colonne complet, $H$ est définie positive. Dans sa base propre, l'erreur par rapport au minimiseur évolue dans chaque direction par le facteur $1-\eta\lambda_i(H)$. Ainsi $0<\eta<2/\lambda_{\max}(H)$ garantit la convergence. En cas de rang déficient, les directions du noyau gardent leur composante initiale ; la même borne assure la convergence des directions de valeur propre positive lorsque $H\ne0$.

Pour inclure un biais, on ajoute une colonne de uns à $X$. La mise à l'échelle des variables peut modifier le conditionnement et la vitesse de convergence. Le code n'effectue pas cette normalisation automatiquement.

## 3. Exercices corrigés

1. Avec $\kappa=2$, $\eta=0,25$, $x_0=2$, déterminer trois itérations. **Correction :** $x_1=1$, $x_2=0,5$, $x_3=0,25$.
2. Pour $\kappa=2$, comparer $\eta=0,75$, $1$, $1,2$. **Correction :** facteurs $-0,5$, $-1$, $-1,4$ ; convergence oscillante, amplitude constante, divergence.
3. Ajuster la droite $y=1+2t$ sur cinq abscisses équidistantes entre -1 et 1, biais inclus. **Correction :** $H=\operatorname{diag}(1,1/2)$ ; $\eta=0,5$ donne des facteurs $0,5$ et $0,75$. Les coefficients convergent vers $(1,2)$.
4. Pourquoi ne peut-on pas utiliser la borne scalaire avec une courbure arbitraire pour tout réseau de neurones ? **Correction :** le spectre de la Hessienne dépend du point et du problème, et la fonction peut être non convexe. Une garantie globale exige d'autres hypothèses.

[Code](../math_for_ai/optimization.py) · [Notebook](../notebooks/01-foundations.ipynb) · [Laboratoire public](https://christ-lowe-portfolio.vercel.app/fr/laboratoire)
