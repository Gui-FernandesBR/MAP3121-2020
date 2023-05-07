# %%
import numpy as np

np.set_printoptions(precision=15)

from sympy import symbols

import matplotlib.pyplot as plt

import math

# %%
# Q6
print("SE QUISER COMPARAR COM UMA FUNÇÃO QUE JÁ SABE QUE A RESPOSTA")
print("TROCAR NA FUNÇÃO def funcao_certa")
print("TEM A BIBLIOTECA MATH")
print("")

x0 = [-1, 1, 2, 3]
f = [(1 / 2) ** (-1), (1 / 2) ** 1, (1 / 2) ** 2, (1 / 2) ** 3]
x = symbols("x")


def main():
    N = newton(x0, f)
    coefN = N[0]
    coefL = lagrange(x0, f)
    print("")
    print("p(x) =", N[1])
    plot(x0, coefN, coefL)


def newton(x0, f):
    coef = []
    for k in range(len(x0) - 1):
        f_d = []
        coef.append(f[0])
        # print('k=',k)
        for i in range(len(f) - 1):
            #   print('i=',i)
            val = (f[i + 1] - f[i]) / (x0[i + 1 + k] - x0[i])
            f_d.append(val)
        #  print('val = ',val)
        f = f_d.copy()
        # print('f =',np.array(f))
    coef.append(f[0])
    print("Seus coeficientes das diferenças divididas: ", np.array(coef))
    L = []
    pol = [1]
    for i in range(len(coef)):
        if i == 0:
            L.append(coef[i])
        elif i == 1:
            L[i - 1] += -1 * coef[i] * x0[i - 1]
            L.append(coef[i])
            pol.append(-x0[i - 1])
        else:
            pol2 = pol.copy()
            pol2.append(0)
            for k in range(len(pol)):
                pol2[k] = 0
            for j in range(len(pol)):
                pol2[j] += pol[j]
                pol2[j + 1] += pol[j] * x0[i - 1] * (-1)
            #         print('j= ',j)
            #        print(pol2)
            pol = pol2.copy()
            for r in range(len(pol)):
                if r == 0:
                    L.append(coef[i])
                else:
                    k = len(L) - 1 - r
                    L[k] += coef[i] * pol[r]
    exp = 0
    for i in range(len(L)):
        exp += (x**i) * np.array(L[i])
    # print('Sua expressão por Newton: p(x)=',exp)
    print("")
    return coef, exp


def lagrange(x0, f):
    L = []
    coef = []
    exp_2 = 0
    for i in range(len(x0)):
        val = 1
        exp = 1
        for k in range(len(x0)):
            if k != i:
                val *= x0[i] - x0[k]
                exp *= x - x0[k]
        val2 = f[i] / val
        coef.append(val2)
        exp = exp * val2
        L.append(val)
        exp_2 += exp
    print("Coeficientes L de lagrange: ", np.array(L))
    # print('Sua expressão por Lagrande: p(x)=',exp_2)
    return coef


def funcao_nos_pontos_N(xi, coef, x0):
    exp = coef[0]
    for i in range(1, len(coef)):
        # print('i=',i)
        exp3 = 1
        for k in range(i):
            #   print('k=',k)
            exp2 = xi - x0[k]
            #  print('exp2 = ',exp2)
            exp3 = exp3 * exp2
        # print('exp3 = ',exp3)
        exp4 = coef[i] * exp3
        # print('exp4 =',exp4)
        exp = exp + exp4
        # print('exp=',exp)
    return exp


def funcao_nos_pontos_L(xi, coef, x0):
    val = 0
    for i in range(len(x0)):
        exp = 1
        for k in range(len(x0)):
            if k != i:
                exp *= xi - x0[k]
        # print(coef)
        exp = exp * coef[i]
        val += exp
    return val


def funcao_certa(xi):
    val = (1 / 2) ** xi
    # print(val)
    return val


def plot(x0, coefN, coefL):
    x_data = np.arange(x0[0] - 1, x0[-1] + 1.1, 0.1)
    # print(x_data)
    y_dataN = []
    y_dataL = []
    y_correct = []
    for i in range(len(x_data)):
        val = funcao_nos_pontos_N(x_data[i], coefN, x0)
        y_dataN.append(val)
        val2 = funcao_certa(x_data[i])
        y_correct.append(val2)
        val3 = funcao_nos_pontos_L(x_data[i], coefL, x0)
        y_dataL.append(val3)
    x00 = []
    x11 = []
    for i in range(len(y_dataN)):
        x00.append(x0[0])
        x11.append(x0[-1])

    plt.plot(x_data, y_dataN, color="Green", label="Interpolação Newton")
    plt.plot(x_data, y_correct, color="Red", label="Função correta")
    plt.plot(x_data, y_dataL, color="Yellow", label="Interpolação lagrange")
    plt.plot(x00, y_dataN)
    plt.scatter(x0, f, color="Black", label="Intersecções")
    plt.plot(x11, y_dataN)
    plt.title("Gráfico função")
    plt.grid()
    plt.legend()
    plt.show()


main()

# %%
(
    -0.0260416666666667 * (((1 / 2) ** (1 / 5)) ** 3)
    + 0.21875 * (((1 / 2) ** (1 / 5)) ** 2)
    - 0.723958333333333 * (((1 / 2) ** (1 / 5)))
    + 1.03125
)

# %%
# Q1

print("SE QUISER COMPARAR COM UMA FUNÇÃO QUE JÁ SABE QUE A RESPOSTA")
print("TROCAR NA FUNÇÃO def funcao_certa")
print("TEM A BIBLIOTECA MATH")
print("")

x0 = [17 / 11, 93 / 55, 101 / 55, 109 / 55, 117 / 55, 25 / 11]
f = [
    1 / ((15 * (17 / 11) / 8) + (4 / 3)),
    1 / ((15 * (93 / 55) / 8) + (4 / 3)),
    1 / ((15 * (101 / 55) / 8) + (4 / 3)),
    1 / ((15 * (109 / 55) / 8) + (4 / 3)),
    1 / ((15 * (117 / 55) / 8) + (4 / 3)),
    1 / ((15 * (25 / 11) / 8) + (4 / 3)),
]
x = symbols("x")


def main():
    N = newton(x0, f)
    coefN = N[0]
    coefL = lagrange(x0, f)
    print("")
    print("p(x) =", N[1])
    plot(x0, coefN, coefL)


def newton(x0, f):
    coef = []
    for k in range(len(x0) - 1):
        f_d = []
        coef.append(f[0])
        # print('k=',k)
        for i in range(len(f) - 1):
            #   print('i=',i)
            val = (f[i + 1] - f[i]) / (x0[i + 1 + k] - x0[i])
            f_d.append(val)
        #  print('val = ',val)
        f = f_d.copy()
        # print('f =',np.array(f))
    coef.append(f[0])
    print("Seus coeficientes das diferenças divididas: ", np.array(coef))
    L = []
    pol = [1]
    for i in range(len(coef)):
        if i == 0:
            L.append(coef[i])
        elif i == 1:
            L[i - 1] += -1 * coef[i] * x0[i - 1]
            L.append(coef[i])
            pol.append(-x0[i - 1])
        else:
            pol2 = pol.copy()
            pol2.append(0)
            for k in range(len(pol)):
                pol2[k] = 0
            for j in range(len(pol)):
                pol2[j] += pol[j]
                pol2[j + 1] += pol[j] * x0[i - 1] * (-1)
            #         print('j= ',j)
            #        print(pol2)
            pol = pol2.copy()
            for r in range(len(pol)):
                if r == 0:
                    L.append(coef[i])
                else:
                    k = len(L) - 1 - r
                    L[k] += coef[i] * pol[r]
    exp = 0
    for i in range(len(L)):
        exp += (x**i) * np.array(L[i])
    # print('Sua expressão por Newton: p(x)=',exp)
    print("")
    return coef, exp


def lagrange(x0, f):
    L = []
    coef = []
    exp_2 = 0
    for i in range(len(x0)):
        val = 1
        exp = 1
        for k in range(len(x0)):
            if k != i:
                val *= x0[i] - x0[k]
                exp *= x - x0[k]
        val2 = f[i] / val
        coef.append(val2)
        exp = exp * val2
        L.append(val)
        exp_2 += exp
    print("Coeficientes L de lagrange: ", np.array(L))
    # print('Sua expressão por Lagrande: p(x)=',exp_2)
    return coef


def funcao_nos_pontos_N(xi, coef, x0):
    exp = coef[0]
    for i in range(1, len(coef)):
        # print('i=',i)
        exp3 = 1
        for k in range(i):
            #   print('k=',k)
            exp2 = xi - x0[k]
            #  print('exp2 = ',exp2)
            exp3 = exp3 * exp2
        # print('exp3 = ',exp3)
        exp4 = coef[i] * exp3
        # print('exp4 =',exp4)
        exp = exp + exp4
        # print('exp=',exp)
    return exp


def funcao_nos_pontos_L(xi, coef, x0):
    val = 0
    for i in range(len(x0)):
        exp = 1
        for k in range(len(x0)):
            if k != i:
                exp *= xi - x0[k]
        # print(coef)
        exp = exp * coef[i]
        val += exp
    return val


def funcao_certa(xi):
    val = xi**4 - 3 * (xi**2) + 1
    # print(val)
    return val


def plot(x0, coefN, coefL):
    x_data = np.arange(x0[0] - 1, x0[-1] + 1.1, 0.1)
    # print(x_data)
    y_dataN = []
    y_dataL = []
    y_correct = []
    for i in range(len(x_data)):
        val = funcao_nos_pontos_N(x_data[i], coefN, x0)
        y_dataN.append(val)
        val2 = funcao_certa(x_data[i])
        y_correct.append(val2)
        val3 = funcao_nos_pontos_L(x_data[i], coefL, x0)
        y_dataL.append(val3)
    x00 = []
    x11 = []
    for i in range(len(y_dataN)):
        x00.append(x0[0])
        x11.append(x0[-1])

    plt.plot(x_data, y_dataN, color="Green", label="Interpolação Newton")
    # plt.plot(x_data,y_correct, color = 'Red', label = 'Função correta')
    plt.plot(x_data, y_dataL, color="Yellow", label="Interpolação lagrange")
    plt.plot(x00, y_dataN)
    plt.scatter(x0, f, color="Black", label="Intersecções")
    plt.plot(x11, y_dataN)
    plt.title("Gráfico função")
    plt.grid()
    plt.legend()
    plt.show()


main()

# %%
# Q2

print("SE QUISER COMPARAR COM UMA FUNÇÃO QUE JÁ SABE QUE A RESPOSTA")
print("TROCAR NA FUNÇÃO def funcao_certa")
print("TEM A BIBLIOTECA MATH")
print("")

x0 = [-10, -8, -6]
f = [1 / 9, 3 / 5, -4 / 5]
x = symbols("x")


def main():
    N = newton(x0, f)
    coefN = N[0]
    coefL = lagrange(x0, f)
    print("")
    print("p(x) =", N[1])
    plot(x0, coefN, coefL)


def newton(x0, f):
    coef = []
    for k in range(len(x0) - 1):
        f_d = []
        coef.append(f[0])
        # print('k=',k)
        for i in range(len(f) - 1):
            #   print('i=',i)
            val = (f[i + 1] - f[i]) / (x0[i + 1 + k] - x0[i])
            f_d.append(val)
        #  print('val = ',val)
        f = f_d.copy()
        # print('f =',np.array(f))
    coef.append(f[0])
    print("Seus coeficientes das diferenças divididas: ", np.array(coef))
    L = []
    pol = [1]
    for i in range(len(coef)):
        if i == 0:
            L.append(coef[i])
        elif i == 1:
            L[i - 1] += -1 * coef[i] * x0[i - 1]
            L.append(coef[i])
            pol.append(-x0[i - 1])
        else:
            pol2 = pol.copy()
            pol2.append(0)
            for k in range(len(pol)):
                pol2[k] = 0
            for j in range(len(pol)):
                pol2[j] += pol[j]
                pol2[j + 1] += pol[j] * x0[i - 1] * (-1)
            #         print('j= ',j)
            #        print(pol2)
            pol = pol2.copy()
            for r in range(len(pol)):
                if r == 0:
                    L.append(coef[i])
                else:
                    k = len(L) - 1 - r
                    L[k] += coef[i] * pol[r]
    exp = 0
    for i in range(len(L)):
        exp += (x**i) * np.array(L[i])
    # print('Sua expressão por Newton: p(x)=',exp)
    print("")
    return coef, exp


def lagrange(x0, f):
    L = []
    coef = []
    exp_2 = 0
    for i in range(len(x0)):
        val = 1
        exp = 1
        for k in range(len(x0)):
            if k != i:
                val *= x0[i] - x0[k]
                exp *= x - x0[k]
        val2 = f[i] / val
        coef.append(val2)
        exp = exp * val2
        L.append(val)
        exp_2 += exp
    print("Coeficientes L de lagrange: ", np.array(L))
    # print('Sua expressão por Lagrande: p(x)=',exp_2)
    return coef


def funcao_nos_pontos_N(xi, coef, x0):
    exp = coef[0]
    for i in range(1, len(coef)):
        # print('i=',i)
        exp3 = 1
        for k in range(i):
            #   print('k=',k)
            exp2 = xi - x0[k]
            #  print('exp2 = ',exp2)
            exp3 = exp3 * exp2
        # print('exp3 = ',exp3)
        exp4 = coef[i] * exp3
        # print('exp4 =',exp4)
        exp = exp + exp4
        # print('exp=',exp)
    return exp


def funcao_nos_pontos_L(xi, coef, x0):
    val = 0
    for i in range(len(x0)):
        exp = 1
        for k in range(len(x0)):
            if k != i:
                exp *= xi - x0[k]
        # print(coef)
        exp = exp * coef[i]
        val += exp
    return val


def funcao_certa(xi):
    val = xi**4 - 3 * (xi**2) + 1
    # print(val)
    return val


def plot(x0, coefN, coefL):
    x_data = np.arange(x0[0] - 1, x0[-1] + 1.1, 0.1)
    # print(x_data)
    y_dataN = []
    y_dataL = []
    y_correct = []
    for i in range(len(x_data)):
        val = funcao_nos_pontos_N(x_data[i], coefN, x0)
        y_dataN.append(val)
        val2 = funcao_certa(x_data[i])
        y_correct.append(val2)
        val3 = funcao_nos_pontos_L(x_data[i], coefL, x0)
        y_dataL.append(val3)
    x00 = []
    x11 = []
    for i in range(len(y_dataN)):
        x00.append(x0[0])
        x11.append(x0[-1])

    plt.plot(x_data, y_dataN, color="Green", label="Interpolação Newton")
    # plt.plot(x_data,y_correct, color = 'Red', label = 'Função correta')
    plt.plot(x_data, y_dataL, color="Yellow", label="Interpolação lagrange")
    plt.plot(x00, y_dataN)
    plt.scatter(x0, f, color="Black", label="Intersecções")
    plt.plot(x11, y_dataN)
    plt.title("Gráfico função")
    plt.grid()
    plt.legend()
    plt.show()


main()

# %%
# Q2-RIC

print("SE QUISER COMPARAR COM UMA FUNÇÃO QUE JÁ SABE QUE A RESPOSTA")
print("TROCAR NA FUNÇÃO def funcao_certa")
print("TEM A BIBLIOTECA MATH")
print("")

x0 = [-10, -8, -6]
f = [1 / 9, 3 / 5, -4 / 5]
x = symbols("x")


def main():
    N = newton(x0, f)
    coefN = N[0]
    coefL = lagrange(x0, f)
    print("")
    print("p(x) =", N[1])
    plot(x0, coefN, coefL)


def newton(x0, f):
    coef = []
    for k in range(len(x0) - 1):
        f_d = []
        coef.append(f[0])
        # print('k=',k)
        for i in range(len(f) - 1):
            #   print('i=',i)
            val = (f[i + 1] - f[i]) / (x0[i + 1 + k] - x0[i])
            f_d.append(val)
        #  print('val = ',val)
        f = f_d.copy()
        # print('f =',np.array(f))
    coef.append(f[0])
    print("Seus coeficientes das diferenças divididas: ", np.array(coef))
    L = []
    pol = [1]
    for i in range(len(coef)):
        if i == 0:
            L.append(coef[i])
        elif i == 1:
            L[i - 1] += -1 * coef[i] * x0[i - 1]
            L.append(coef[i])
            pol.append(-x0[i - 1])
        else:
            pol2 = pol.copy()
            pol2.append(0)
            for k in range(len(pol)):
                pol2[k] = 0
            for j in range(len(pol)):
                pol2[j] += pol[j]
                pol2[j + 1] += pol[j] * x0[i - 1] * (-1)
            #         print('j= ',j)
            #        print(pol2)
            pol = pol2.copy()
            for r in range(len(pol)):
                if r == 0:
                    L.append(coef[i])
                else:
                    k = len(L) - 1 - r
                    L[k] += coef[i] * pol[r]
    exp = 0
    for i in range(len(L)):
        exp += (x**i) * np.array(L[i])
    # print('Sua expressão por Newton: p(x)=',exp)
    print("")
    return coef, exp


def lagrange(x0, f):
    L = []
    coef = []
    exp_2 = 0
    for i in range(len(x0)):
        val = 1
        exp = 1
        for k in range(len(x0)):
            if k != i:
                val *= x0[i] - x0[k]
                exp *= x - x0[k]
        val2 = f[i] / val
        coef.append(val2)
        exp = exp * val2
        L.append(val)
        exp_2 += exp
    print("Coeficientes L de lagrange: ", np.array(L))
    # print('Sua expressão por Lagrande: p(x)=',exp_2)
    return coef


def funcao_nos_pontos_N(xi, coef, x0):
    exp = coef[0]
    for i in range(1, len(coef)):
        # print('i=',i)
        exp3 = 1
        for k in range(i):
            #   print('k=',k)
            exp2 = xi - x0[k]
            #  print('exp2 = ',exp2)
            exp3 = exp3 * exp2
        # print('exp3 = ',exp3)
        exp4 = coef[i] * exp3
        # print('exp4 =',exp4)
        exp = exp + exp4
        # print('exp=',exp)
    return exp


def funcao_nos_pontos_L(xi, coef, x0):
    val = 0
    for i in range(len(x0)):
        exp = 1
        for k in range(len(x0)):
            if k != i:
                exp *= xi - x0[k]
        # print(coef)
        exp = exp * coef[i]
        val += exp
    return val


def funcao_certa(xi):
    val = xi**4 - 3 * (xi**2) + 1
    # print(val)
    return val


def plot(x0, coefN, coefL):
    x_data = np.arange(x0[0] - 1, x0[-1] + 1.1, 0.1)
    # print(x_data)
    y_dataN = []
    y_dataL = []
    y_correct = []
    for i in range(len(x_data)):
        val = funcao_nos_pontos_N(x_data[i], coefN, x0)
        y_dataN.append(val)
        val2 = funcao_certa(x_data[i])
        y_correct.append(val2)
        val3 = funcao_nos_pontos_L(x_data[i], coefL, x0)
        y_dataL.append(val3)
    x00 = []
    x11 = []
    for i in range(len(y_dataN)):
        x00.append(x0[0])
        x11.append(x0[-1])

    plt.plot(x_data, y_dataN, color="Green", label="Interpolação Newton")
    # plt.plot(x_data,y_correct, color = 'Red', label = 'Função correta')
    plt.plot(x_data, y_dataL, color="Yellow", label="Interpolação lagrange")
    plt.plot(x00, y_dataN)
    plt.scatter(x0, f, color="Black", label="Intersecções")
    plt.plot(x11, y_dataN)
    plt.title("Gráfico função")
    plt.grid()
    plt.legend()
    plt.show()


main()
