from sympy.tensor import IndexedBase
from sympy import *


def run():
    # M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
    C = MatrixSymbol("C", 3, 3)
    D = Matrix(C)
    i = IndexedBase("i")
    j = IndexedBase("j")
    k = IndexedBase("k")
    l = IndexedBase("l") 
    m = IndexedBase("m") 

    derivative = diff(D.det(), D)

    print(simplify(derivative))

    print(simplify(diff(  C[i, j]*C[j, k], C[l, m])) )

if __name__ == '__main__':
    run()
