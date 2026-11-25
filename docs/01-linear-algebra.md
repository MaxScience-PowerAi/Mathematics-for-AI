# 01 — Algèbre linéaire pour l'IA

Objectifs : manipuler les vecteurs, mesurer la similarité, projeter un vecteur, résoudre un problème de moindres carrés et comprendre une approximation de faible rang. Les calculs sont effectués dans des espaces euclidiens réels de dimension finie.

## 1. Produit scalaire et similarité cosinus

Pour $x,y\in\mathbb R^d$, $\langle x,y\rangle=\sum_i x_i y_i$ et $\|x\|_2=\sqrt{\langle x,x\rangle}$. Cauchy–Schwarz donne $|\langle x,y\rangle|\le\|x\|_2\|y\|_2$.

Si les deux vecteurs sont non nuls :

$$s(x,y)=\frac{x^Ty}{\|x\|_2\|y\|_2}\in[-1,1].$$

Le score est invariant par multiplication de chaque vecteur par un scalaire strictement positif. Il mesure l'orientation, pas l'égalité des normes. Une valeur élevée peut être utile pour comparer des embeddings, mais ne garantit pas une similarité sémantique : celle-ci dépend du modèle d'embedding.

## 2. Projection orthogonale

Soit $u\ne0$. Cherchons $p=\alpha u$ minimisant $\|x-\alpha u\|_2^2$. La dérivée par rapport à $\alpha$ vaut $-2x^Tu+2\alpha u^Tu$, donc

$$p=\frac{x^Tu}{u^Tu}u.$$

Le résidu est orthogonal à $u$, car $(x-p)^Tu=0$. Cette propriété est vérifiée dans les tests. La méthode normalise d'abord $u$ pour limiter les problèmes d'échelle.

## 3. Moindres carrés

Pour $A\in\mathbb R^{n\times d}$ et $b\in\mathbb R^n$, cherchons un minimiseur de $F(w)=\tfrac12\|Aw-b\|_2^2$. Son gradient est $A^T(Aw-b)$, donc tout minimiseur satisfait les équations normales $A^TAw=A^Tb$. La fonction est convexe. Si $A$ possède un rang colonne complet, le minimiseur est unique. Dans le cas contraire, plusieurs solutions peuvent exister ; numpy.linalg.lstsq retourne une solution de norme minimale avec son seuil numérique de rang.

On évite de calculer explicitement $(A^TA)^{-1}$ : les équations normales peuvent amplifier les problèmes de conditionnement.

## 4. Décomposition en valeurs singulières

La SVD écrit $A=U\Sigma V^T$, avec valeurs singulières $\sigma_1\ge\cdots\ge\sigma_k\ge0$, $k=\min(n,d)$. La matrice tronquée

$$A_r=\sum_{i=1}^r\sigma_i u_i v_i^T$$

minimise l'erreur de Frobenius parmi les matrices de rang au plus $r$ (théorème d'Eckart–Young–Mirsky). En particulier, $\|A-A_r\|_F^2=\sum_{i>r}\sigma_i^2$. Cette propriété motive la compression et la réduction de dimension. La PCA est liée à la SVD des données centrées ; ce dépôt ne propose pas encore une implémentation PCA complète.

## 5. Exercices corrigés

1. Calculer la similarité de $x=(1,2)$ et $y=(2,1)$. **Correction :** produit scalaire 4, normes $\sqrt5$ ; résultat $4/5=0,8$.
2. Projeter $(3,2)$ sur $(1,1)$. **Correction :** $(5/2,5/2)$ ; le résidu $(1/2,-1/2)$ est orthogonal à $(1,1)$.
3. Ajuster $y=a+bt$ aux points $(0,1),(1,3),(2,5)$. **Correction :** $a=1,b=2$ ; le résidu est nul et la matrice de conception a rang 2.
4. Calculer l'erreur de rang 1 de $A=\operatorname{diag}(5,3,1)$. **Correction :** on conserve 5 ; l'erreur de Frobenius vaut $\sqrt{3^2+1^2}=\sqrt{10}$.

Expérimente avec [le notebook](../notebooks/01-foundations.ipynb) et [les tests](../tests/test_foundations.py). Modifie les données, vérifie les formules et explique tes résultats avant de les publier.
