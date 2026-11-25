"""Run from repository root: python -m examples.run_demo"""
from math_for_ai import cosine_similarity, quadratic_descent

if __name__ == '__main__':
    print('Cosine similarity:', cosine_similarity([1, 2], [2, 1]))
    x, f = quadratic_descent(2, .25, 2, 6)
    for i, (a, b) in enumerate(zip(x, f)):
        print(f'{i:2d} x={a:.6f} f(x)={b:.6f}')
