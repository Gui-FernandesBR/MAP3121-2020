# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:48:38 2020

@author: Guilherme Fernandes Alves
"""


def funcao(x):
    """Funcao que recebe um valor "x" e calcula f(x) = x² - 2"""
    return x * x * x - 163 * x * x + 5298 * x - 22368


def derivada(x):
    """Funcao que recebe um valor "x" e calcula f(x) = 2*x"""
    return 3 * x * x - 326 * x + 5298


erro = 0.010  # 10**(-3)
prec = 1
x = 0

while prec > erro:
    x = x - (funcao(x)) / (derivada(x))
    prec = funcao(x)

print("Raiz é igual a ", x, "com precisao igual a", prec)
# print("A diferença entre o valor encontrado e o valor real é de ", x - (2**0.5))
