# -*- coding: utf-8 -*-
"""
ESCOLA POLITÉCNICA DA UNIVERSIDADE DE SÃO PAULO
EXERCÍCIO PROGRAMA 2 PARA A DISCIPLINA MAP3121 (MÉTODOS NUMÉRICOS)
- Junho de 2020, São Paulo, 

@authors: Guilherme Fernandes Alves e Ricardo Aguiar de Andrade
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random

np.set_printoptions(precision=2)  # Limita precisão de prints do Numpy


def main_mmq():
    """Resolve o problema dos mínimos quadrados desse EP, imprime os resultados
    de intensidades calculadas (tal como o erro quadrático associado) de acordo
    com o teste escolhido pelo usuário
    """
    global n  # Número de partições no espaço

    t = 1  # Esse é o nosso tempo total (T)
    teste = input("Indique qual teste você deseja realizar ('a', 'b', 'c' ou 'd'): ")

    if teste != "a" and teste != "b" and teste != "c" and teste != "d":
        s = "Modo selecionado inválido, verifique se você digitou corretamente"
        print(s)
        main_mmq()
    if teste == "a" or teste == "b":
        n = 128
    else:
        n = int(input("Digite o valor de N: "))
    m = n  # Condição desse EP2, para o método de Crank-Nicolson

    gr = input("Você deseja plotar os gráficos das soluções? ('s'=sim e 'n'=não): ")
    if gr != "s" and gr != "n":
        print("Digite 's' para 'sim' ou 'n' não 'não'.")
        main_mmq()
    elif gr == "s":
        gr = True
    elif gr == "n":
        gr = False

    posic = posicao_fontes(teste)  # Vetor com as posições das fontes
    nf = len(posic)  # Número de fontes

    M = Matrix(nf, m, t, posic, teste, gr).copy()
    if nf == 1:
        intens = (M[2][0]) / (M[0][0])  # Para nf=1, a solução trivial!
        print("Valores das intensidades ak = ", intens)
        print()
        print("Erro quadrático = ", Erro_quad(nf, m, t, posic, teste, intens, gr))
        print()
        Recupera(intens, teste, gr, m, t, posic, nf)

    else:
        vet = decomp_sim(M[0].copy())
        intens = solve_sys2(M[1].copy(), vet[0], vet[1], M[2].copy(), nf)
        print("Valores das intensidades ak = ", np.array(intens))
        print()
        print("Erro quadrático = ", Erro_quad(nf, m, t, posic, teste, intens, gr))
        print()
        Recupera(intens, teste, gr, m, t, posic, nf)


def posicao_fontes(teste):
    """Função que retorna um vetor com as posições para as fontes pontuais do
    problema, dependendo do tipo de teste escolhido pelo usuário

    Arguments:
        teste {str} -- Variável utilizada para indicar qual o tipo de teste a ser realizado

    Returns:
        list -- Vetor contendo as posições das fontes pontuais do exercicio, de
                acordo  com o tipo de teste a ser realizado
    """
    if teste == "a":
        posi = [0.35]
    elif teste == "b":
        posi = [0.15, 0.3, 0.7, 0.8]
    elif (
        teste == "c" or teste == "d"
    ):  # Valores retirados do arquivo .txt de testes do EP2
        posi = [
            0.14999999999999999,
            0.20000000000000001,
            0.29999999999999999,
            0.34999999999999998,
            0.50000000000000000,
            0.59999999999999998,
            0.69999999999999996,
            0.72999999999999998,
            0.84999999999999998,
            0.90000000000000002,
        ]
    return posi


def produto_interno(u, v):
    """Função que recebe dois vetores de tamanho N-1 e retorna o produto interno
    entre esses dois vetores, tal como explicado no enunciado do EP. Note que os
    extremos (fronteiras) do vetor não etram na conta.

    Arguments:
        u {list} -- Uma lista de valores (vetor) de tamanho N-1
        v {list} -- Uma lista de valores (vetor) de tamanho N-1

    Returns:
        float -- Número real resultado da operação produto interno
    """
    q = len(u)
    if q != len(v):
        print(
            "ALERTA: Para o cálculo do produto interno, os 2 vetores devem ser de mesmo tamanho, corrija o problema antes de prosseguir"
        )
        return None

    soma = 0
    for i in range(1, q - 1):
        soma = soma + u[i] * v[i]

    return soma


def r(ti):
    """Função que descreve a variação temporal das fontes pontuais, tal como
    especificado no enunciado do EP.

    Arguments:
        ti {float} -- Instante de tempo para o qual se deseja saber o r(ti)

    Returns:
        float -- valor de r(t) calculada no instante desejado
    """
    p = 10 * (1 + math.cos(5 * ti))
    return p


def decomp_sim(vet):
    """Função que recebe uma matriz quadrada A e realiza a decomposição A = LDL^t.
    - A matriz L é triangular inferior, tendo sua diagonal principal preenchida com 1s
    - A matriz D é diagonal.
    - Note que a única condição sobre A é que ela seja quadrada, ou seja, que o
    número de linhas seja igual ao número de colunas.

    Arguments:
        list -- Uma lista de listas (matriz) a ser decomposta.
                - O numero de linhas deve ser igual ao numero de colunas!

    Returns:
        list -- Uma lista contendo duas listas:
                - A primeira é a matriz L, triangular inferior com diagonal preenchida por 1s
                - A segunda é a matriz D diagonal
    """

    li = [0]
    comp = len(vet)
    L = li * comp
    D = li * comp
    v = li * comp

    for i in range(comp):
        L[i] = li * comp
        D[i] = li * comp

    for i in range(comp):
        L[i][i] = 1

    for i in range(comp):  # Step1
        for j in range(i):  # Step2
            v[j] = L[i][j] * D[j][j]

        somatorio = 0  # Step3
        for j in range(i):
            somatorio += L[i][j] * v[j]
        D[i][i] = vet[i][i] - somatorio

        for j in range(i + 1, comp):  # Step4
            somat = 0
            for k in range(i):
                somat += L[j][k] * v[k]
            L[j][i] = (vet[j][i] - somat) / D[i][i]

    return [L, D]


def decomp(vet1, vet2):  # Cópia do EP1
    """Função "decomp". Recebe dois vetores que representam a diagonal principal e subdiagonal de uma matriz A
    tridiagonal simétrica e retorna outros dois vetores que representam as matrizes D e L que respeitam
    a seguinte relação: A = LDL^t
    Obs.: Essa função foi copiada do EP1 da dupla e será utilizada para resolver
    o método de Crank-Nicolson quando necessário"""

    vet_d = crie_vetor(len(vet1))
    vet_l = crie_vetor(len(vet2))

    vet_d[0] = vet1[0]

    h = 0
    while h < n - 2:
        vet_l[h] = vet2[h] / vet_d[h]
        vet_d[h + 1] = vet1[h + 1] - (vet_l[h] ** 2) * vet_d[h]

        h += 1

    return [vet_d, vet_l]


def crie_vetor(n_colunas):  # Cópia do EP1
    """Função "crie_vetor". Recebe o parâmetro "n_colunas" retornando um vetor
    nulo de dimensão (comprimento) n-1. Obs.: Função copiada do EP1 da dupla."""
    vetor = []
    for j in range(n_colunas):
        x = 0
        vetor.append(x)
        j = j  # Apenas para que não dê problema de variácel inutilizada
    return vetor


def solve_sys(unew, vet_l, vet_d, vet_dir):  # Funcão idêntica à utilizada no EP1
    """Função "solve_sys". Recebe 4 vetores que caracterizam um sistema do tipo:
    LDL^t x = b, onde: 'vet_l' representa a matriz L; 'vet_d' representa a matriz D; 'vet_dir' representa
    a matriz b e 'unew' representa a matriz x. Retorna o vetor solução do sistema 'unew'.
    """
    y = crie_vetor(n - 1)  # Resolução do sistema Ly = b
    y[0] = vet_dir[1]
    for i in range(1, n - 1):
        y[i] = vet_dir[i + 1] - vet_l[i - 1] * y[i - 1]
    z = crie_vetor(n - 1)  # Resolução do sistema Dz = y
    for o in range(n - 1):
        z[o] = y[o] / vet_d[o]
    unew[n - 1] = z[n - 2]
    for q in range(n - 2, 0, -1):  # Resolução do sistema L^t x = z
        unew[q] = z[q - 1] - vet_l[q - 1] * unew[q + 1]
    return unew


def solve_sys2(sol, vet_l, vet_d, vet_dir, nf):
    """Função que resolve um sistema do tipo: LDL^t x = b
    Obs.: Essa função dará conta de resolver o sistema normal e entregar a
    solução do problema de mínimos quadrados. Essa função foi desenvolvida
    exclusivamente para o EP2.

    Arguments:
        sol {list} -- Uma lista de comprimento nf contendo as intensidades das fontes
        vet_l {list} -- Uma lista de listas que representa a matriz triangular inferior (L)
        vet_d {list} -- Uma lista de listas que representa a matriz diagonal (D)
        vet_dir {list} -- Uma lista de comprimento nf representando o lado direito da equação
        nf {int} -- Número de fontes utilizados no problema em questão

    Returns:
        list -- Uma lista de comprimento nf contendo as intensidades das fontes
    """

    y = crie_vetor(nf)  # Resolução do sistema Ly = b
    y[0] = vet_dir[0]

    for i in range(1, nf):
        soma = 0
        for j in range(0, i):
            soma += vet_l[i][j] * y[j]
        y[i] = vet_dir[i] - soma

    z = crie_vetor(nf)  # Resolução do sistema Dz = y
    for o in range(nf):
        z[o] = y[o] / vet_d[o][o]

    # Vamos transpor a matriz L
    vet_t = []
    for i in range(nf):
        linha = []
        for j in range(nf):
            linha.append(0)
        vet_t.append(linha)

    for i in range(nf):
        for j in range(nf):
            vet_t[i][j] = vet_l[j][i]

    sol[nf - 1] = z[nf - 1]
    for q in range(nf - 2, -1, -1):  # Resolução do sistema L^t x = z
        soma = 0
        for r in range(q + 1, nf, 1):
            soma += vet_t[q][r] * sol[r]
        sol[q] = z[q] - soma

    return sol


def fonte_total(x, ti, posic):  # (essa vem do EP1 mas com várias modificações)
    """Função para calcular o calor gerado por TODAS as fontes em uma posição
    específica da barra, para um determinado instante de tempo. Ideal para a
    determinação dos vetores U_{T}

    Arguments:
        x {float}    -- Posição da barra onde queremos calcular o calor gerado
                        pelas fontes, pode variar de 0 até 1
        ti {aloat}   -- Instante de tempo, podendo variar de 0 até t (t=1)
        posic {list} -- Vetor contendo a posição de CADA fonte pontual de calor

    Returns:
        float -- Calor gerado por TODAS as fontes na posição x, instante ti
    """
    delta_x = 1 / n  # delta_x faz o papel do h
    g_total = 0
    for p in posic:
        if p + (delta_x / 2) >= x >= p - (delta_x / 2):
            g_total += 1 / delta_x
    return r(ti) * g_total


def fonte_solo(x, ti, posic, f):
    """Função para calcular o calor gerado por uma ÚNICA fonte pontual de calor
    para uma posição específica em um determinado instante de tempo. Essa função
    será utilizada para a determinação dos vetores u_{k}.

    Arguments:
        x {float}    -- Posição da barra onde queremos calcular o calor gerado
                        pelas fontes, pode variar de 0 até 1
        ti {floar}   -- Instante de tempo, podendo variar de 0 até t (t=1)
        posic {list} -- Vetor contendo a posição de CADA fonte pontual de calor
        f {int}      -- Número da fonte que será utilizada para cálculo, pode
                        variar de 0 até nf-1

    Returns:
        float -- Calor gerado pela fonte de número f+1 na posição x, instante ti
    """
    delta_x = 1 / n  # delta_x faz o papel do h
    p = posic[f]
    g = 0
    if p + (delta_x / 2) >= x >= p - (delta_x / 2):
        g += 1 / delta_x
    return r(ti) * g


def u(vetor, m, t, posic):  # Funcao u convencional, cópia do EP1 com adaptações
    """Função para calcular e retornar, através do método de Crank-Nicolson, o
    vetor de resultados aproximados para o instante tM (t=1). Aqui estamos
    interessados em calcular o vetor U no último instante de tempo, igualzinho
    já fizemos no EP1. OBS.: Aqui consideramos TODAS as fontes atuando na barra

    Arguments:
        vetor {list} -- Vetor que contém a condição de contorno u0 (1a linha)
        m {int}      -- Número de passos no tempo, será igual a 'n' nesse EP
        t {float}    -- Tempo total (T)
        posic {list} -- Vetor contendo a posição de CADA fonte pontual de calor

    Returns:
        list -- vetor contendo a temperatura final em cada posição da barra,
                também conhecido como U_{T} ou Ut
    """
    uold = vetor.copy()
    unew = crie_vetor(len(vetor))
    delta_x = 1 / n
    delta_t = t / m
    lda = n  # Devido à condição (Delta t = Delta x)

    vet1 = []
    for i in range(2, len(vetor)):
        vet1.append(1 + lda)
        i = i

    vet2 = []
    for j in range(2, len(vetor) - 1):
        vet2.append(-lda / 2)
        j = j

    vet_decomp = decomp(vet1, vet2)

    vet_d = vet_decomp[0]
    vet_l = vet_decomp[1]

    vet_dir = crie_vetor(n + 1)  # Início do método de Crank-Nicolson implícito

    for k in range(m):
        for d in range(
            1, n
        ):  # Preenchimento da matriz coluna b para o método de Crank-Nicolson
            if d == 1:
                vet_dir[d] = (
                    uold[d]
                    + (lda / 2) * (uold[d - 1] - 2 * uold[d] + uold[d + 1])
                    + (delta_t / 2)
                    * (
                        fonte_total(d * delta_x, (k + 1) * delta_t, posic)
                        + fonte_total(d * delta_x, k * delta_t, posic)
                    )
                )
            elif d == n - 1:
                vet_dir[d] = (
                    uold[d]
                    + (lda / 2) * (uold[d - 1] - 2 * uold[d] + uold[d + 1])
                    + (delta_t / 2)
                    * (
                        fonte_total(d * delta_x, (k + 1) * delta_t, posic)
                        + fonte_total(d * delta_x, k * delta_t, posic)
                    )
                )
            else:
                vet_dir[d] = (
                    uold[d]
                    + (lda / 2) * (uold[d - 1] - 2 * uold[d] + uold[d + 1])
                    + (delta_t / 2)
                    * (
                        fonte_total(d * delta_x, (k + 1) * delta_t, posic)
                        + fonte_total(d * delta_x, k * delta_t, posic)
                    )
                )

        unew = solve_sys(unew, vet_l, vet_d, vet_dir)

        # Preenchimento da primeira e última posição (fronteiras).
        unew[0] = 0
        unew[n] = 0
        uold = np.array(unew.copy())

    return unew


def u_fonte(vetor, m, t, posic, f):  # Funcao u para fonte única
    """Função para calcular e retornar, através do método de Crank-Nicolson, o
    vetor de resultados aproximados para o instante tM (t=1). Aqui estamos
    interessados em calcular o vetor Uk no último instante de tempo considerando
    uma única fonte atuando na barra. OBS.: Aqui consideramos uma ÚNICA fonte
    atuando na barra.
    Arguments:
        vetor {list} -- Vetor que contém a condição de contorno u0 (1a linha)
        m {int}      -- Número de passos no tempo, será igual a 'n' nesse EP
        t {float}    -- Tempo total (T)
        posic {list} -- Vetor contendo a posição de CADA fonte pontual de calor
        f {int}      -- Número da fonte que será utilizada para cálculo, pode
                        variar de 0 até nf-1

    Returns:
        list -- vetor contendo a temperatura final em cada posição da barra,
                também conhecido como u_{k} ou uk
    """

    uold = vetor.copy()
    unew = crie_vetor(len(vetor))
    delta_x = 1 / n
    delta_t = t / m
    lda = n  # Devido à condição (Delta t = Delta x)

    vet1 = []
    for i in range(2, len(vetor)):
        vet1.append(1 + lda)
        i = i

    vet2 = []
    for j in range(2, len(vetor) - 1):
        vet2.append(-lda / 2)
        j = j

    vet_decomp = decomp(vet1, vet2)

    vet_d = vet_decomp[0]
    vet_l = vet_decomp[1]

    vet_dir = crie_vetor(n + 1)  # Início do método de Crank-Nicolson implícito

    for k in range(m):
        for d in range(
            1, n
        ):  # Preenchimento da matriz coluna b para o método de Crank-Nicolson
            if d == 1:
                vet_dir[d] = (
                    uold[d]
                    + (lda / 2) * (uold[d - 1] - 2 * uold[d] + uold[d + 1])
                    + (delta_t / 2)
                    * (
                        fonte_solo(d * delta_x, (k + 1) * delta_t, posic, f)
                        + fonte_solo(d * delta_x, k * delta_t, posic, f)
                    )
                )
            elif d == n - 1:
                vet_dir[d] = (
                    uold[d]
                    + (lda / 2) * (uold[d - 1] - 2 * uold[d] + uold[d + 1])
                    + (delta_t / 2)
                    * (
                        fonte_solo(d * delta_x, (k + 1) * delta_t, posic, f)
                        + fonte_solo(d * delta_x, k * delta_t, posic, f)
                    )
                )
            else:
                vet_dir[d] = (
                    uold[d]
                    + (lda / 2) * (uold[d - 1] - 2 * uold[d] + uold[d + 1])
                    + (delta_t / 2)
                    * (
                        fonte_solo(d * delta_x, (k + 1) * delta_t, posic, f)
                        + fonte_solo(d * delta_x, k * delta_t, posic, f)
                    )
                )

        unew = solve_sys(unew, vet_l, vet_d, vet_dir)

        # Preenchimento da primeira e última posição (fronteiras).
        unew[0] = 0
        unew[n] = 0
        uold = np.array(unew.copy())

    return unew


def cria_uk(nf, m, t, posic, gr):  # Aqui acaba a tarefa a) do EP2
    """Função que calcula e retorna um vetor contendo cada vetor uk no instante
    final de tempo, resolvendo assim a tarefa a) deste EP2.

    Arguments:
        nf {int}     -- Número de fontes
        m {int}      -- Número de passos no tempo, será igual a 'n' nesse EP
        t {float}    -- Tempo total (T), será igual a 1 nesse EP
        posic {list} -- Vetor contendo a posição de CADA fonte pontual de calor
        gr {bool}    -- Indica se o usuário deseja ou não imprimir os gráficos

    Returns:
        list -- Uma lista de listas (matriz). Cada lista (linha) é um vetor
        contendo valores de temperatura nas diferentes posições da barra, no
        instante t considerando uma única fonte atuando. Ou seja, cada linha da
        nossa matriz será um vetor uk.
    """
    lista = []
    for i in range(nf):
        vet = crie_vetor(n + 1)
        u_ft = u_fonte(vet, m, t, posic, i)
        lista.append(u_ft)

    if gr:
        for i in range(nf):
            pos = []
            for j in range(n + 1):
                pos.append((1 / n) * j)
            plt.plot(pos, lista[i], label="f={0}".format(i + 1))
            plt.xlabel("position")
            plt.ylabel("temperature")
            plt.title("Vetores Uk - Temperatura x posição - Tempo = 1")
            plt.legend()
        plt.grid(True)
        plt.show()
    return lista


def le_matriz():
    """Função que coleta os dados de temperatura na barra para o intante T=1 a
    partir do arquivo "Arquivo teste para o EP2.txt" disponibilizado pelos
    professores para a realização do EP2. Mantenha o arquivo na mesma pasta do
    código!!!

    Returns:
        list -- Vetor contendo os valores de temperatura medida em cada ponto da
        barra para uma malha discretizada com N=2048
    """
    arquivo = open("Arquivo teste para o EP2.txt", "r")  # Modo de leitura
    matriz = []
    i = 0
    for linha in arquivo:
        l = linha.strip()
        if i > 0:
            matriz.append(float(l))
        i += 1
    arquivo.close()
    return matriz


def U_tx(teste, nf, m, t, posic, gr):
    """Calcula e retorna o vetor de temperaturas no instante final da barra,
    dependendo do tipo de teste a ser realizado. Geralmente o vetor de temperaturas
    finais é uma combinação linear dos vetores uk, sendo as intensidades de fonte
    as constantes desse combinação.

    Arguments:
        teste {str}  -- Tipo de teste a ser realizado ('a', 'b, 'c' ou 'd')
        nf {int}     -- Número de fontes
        m {[int}     -- Número de passos no tempo, será igual a 'n' nesse EP
        t {float}    -- Tempo total (T), será igual a 1 nesse EP
        posic {list} -- Vetor contendo a posição de CADA fonte pontual de calor
        gr {bool}    -- Indica se o usuário deseja ou não imprimir os gráficos

    Returns:
        list -- Vetor de temperaturas finais nas diferentes posições da barra, a
        depender do tipo de teste realizado pelo usuário
    """
    if teste == "a":
        vet = crie_vetor(n + 1)
        v = u(vet, m, t, posic)
        for i in range(len(v)):
            v[i] = 7 * v[i]
        if gr:
            plot(
                v,
                "Vetor U(x) calculado pelo método CN",
                "Teste A, Temperature X Position, time=1, (N = 128)",
            )
        return v
    elif teste == "b":
        if gr:
            gr = False
            v = cria_uk(nf, m, t, posic, gr)
            gr = True
        else:
            v = cria_uk(nf, m, t, posic, gr)
        w = []
        for i in range(n + 1):
            w.append(2.3 * v[0][i] + 3.7 * v[1][i] + 0.3 * v[2][i] + 4.2 * v[3][i])
        if gr:
            plot(
                w,
                "Vetor U(x) calculado pelo método CN",
                "Teste B, Temperature X Position, time=1, (N = 128)",
            )
        return w
    elif teste == "c" or teste == "d":
        w = le_matriz()
        stp = int(2048 / n)
        z = []
        for k in range(0, 2049, stp):
            z.append(w[k])

        if teste == "c":
            if gr:
                plot(
                    z,
                    "Vetor U(x) extraído do arquivo",
                    "Teste C, Temperature X Position, time=1, (N = %d)" % (n),
                )
            return z
        else:  # Vamos aplicar um ruído para essa função
            ra = random.random()
            ra = ra - 0.5
            ra = ra * 2
            eps = 0.01
            for i in range(len(z)):
                z[i] = z[i] * (1 + ra * eps)
            if gr:
                plot(
                    z,
                    "Vetor U(x) extraído do arquivo, com ruído",
                    "Teste D, Temperature X Position, time=1, (N = %d)" % (n),
                )
            return z


def Matrix(nf, m, t, posic, teste, gr):
    """Função para determinar as três matrizes do sistema normal, equação 40 do
    enunciado do EP2 ou equação 11 do relatório. *Números sujeitos a alterações

    Arguments:
        nf {int}     -- Número de fontes
        m {int}      -- Número de passos no tempo, será igual a 'n' nesse EP
        t {float}    -- Tempo total (T), será igual a 1 nesse EP
        posic {list} -- Vetor contendo a posição de CADA fonte pontual de calor
        teste {str}  -- Tipo de teste a ser realizado ('a', 'b, 'c' ou 'd')
        gr {bool}    -- Indica se o usuário deseja ou não imprimir os gráficos

    Returns:
        list -- Uma lista contendo 3 listas:
                - A primeira é a matriz simétrica de dimensão nf que fica mais à
                esquerda da igualdade
                - A segunda é um vetor de parâmetros unitários onde serão salvos
                os valores das intensidades ak
                - A terceira é a matriz de dimensão (nf X 1) que fica à direita
                da igualdade
    """

    Ma = []
    uk = cria_uk(nf, m, t, posic, gr).copy()  # Cuidado, isso aqui é uma matriz!!
    ut = U_tx(teste, nf, m, t, posic, gr).copy()

    for i in range(nf):
        linha = []
        for j in range(nf):
            u = uk[i].copy()
            v = uk[j].copy()
            a = produto_interno(u, v).copy()
            linha.append(a)
        Ma.append(linha)  # Ma é a matriz do lado esquerdo

    variavel = []
    for i in range(nf):
        variavel.append(
            1
        )  # Daqui deve sair a matriz de variáveis (constantes de intensidade) do sistema

    Ba = []
    for k in range(nf):
        c = produto_interno(ut, uk[k])  # ut tá com 128 itens enquanto uk tá com 129
        Ba.append(c)  # Ba é a matriz do lado direito do sistema
    return [Ma, variavel, Ba]


def Erro_quad(nf, m, t, posic, teste, intens, gr):
    """Função para calcular o erro quadrático da nossa solução para o problema
    de mínimos quadrados.

    Arguments:
        nf {int}      -- Número de fontes
        m {int}       -- Número de passos no tempo, será igual a 'n' nesse EP
        t {float}     -- Tempo total (T), será igual a 1 nesse EP
        posic {list}  -- Vetor contendo a posição de CADA fonte pontual de calor
        teste {str}   -- Tipo de teste a ser realizado ('a', 'b, 'c' ou 'd')
        intens {list} -- Vetor com a solução final das intensidades das fontes
        gr {bool}     -- Indica se o usuário deseja ou não imprimir os gráficos

    Returns:
        float -- Erro quadrático calculado
    """
    gr = False  # Para não imprimir o mesmo gráfico 2 vezes
    ut = U_tx(teste, nf, m, t, posic, gr)
    uk = cria_uk(nf, m, t, posic, gr)
    delta_x = 1 / n

    somat = 0
    for i in range(
        1, n
    ):  # Não incluindo a fronteira, que será calculada mais abaixo através de integral aproximada por trapézios
        somat2 = 0
        for j in range(nf):
            a = intens[j] * uk[j][i]
            somat2 += a

        desv = (ut[i] - somat2) ** 2
        somat += desv
    E2 = delta_x * somat
    E = E2 ** (0.5)

    """Rotina ainda a ser implementada: método dos trapézios para cálculo do
    erro nas fronteiras -   EM CONSTRUÇÃO!!!"""
    return E


def Recupera(intens, teste, gr, m, t, posic, nf):
    """Após calcular as intensidades das fontes, vamos recalcular o vetor de
    temperaturas finais e então imprimir seu gráfico novamente.

    Args:
        intens (list): Vetor com a solução final das intensidades das fontes
        teste (str)  : Tipo de teste a ser realizado ('a', 'b, 'c' ou 'd')
        gr (bool)    : Indica se o usuário deseja ou não imprimir os gráficos
        m (int)      : Número de passos no tempo, será igual a 'n' nesse EP
        t (float)    : Tempo total (T), será igual a 1 nesse EP
        posic (list) : Vetor contendo a posição de CADA fonte pontual de calor
        nf ([type])  : Número de fontes
    """
    if teste == "a":
        vet = crie_vetor(n + 1)
        v = u(vet, m, t, posic)
        for i in range(len(v)):
            v[i] = (intens[0]) * v[i]
        if gr:
            plot(
                v,
                "Vetor U(x) intensidades recuperadas",
                "Teste A, Temperatura X Posição, time=1, (N = 128)",
            )

    elif teste == "b":
        if gr:
            gr = False
            v = cria_uk(nf, m, t, posic, gr)
            gr = True
        else:
            v = cria_uk(nf, m, t, posic, gr)
        w = []
        for i in range(n + 1):
            w.append(
                intens[0] * v[0][i]
                + intens[1] * v[1][i]
                + intens[2] * v[2][i]
                + intens[3] * v[3][i]
            )
        if gr:
            plot(
                w,
                "Vetor U(x) intensidades recuperada",
                "Teste B, Temperatura X Posição, time=1, (N = 128)",
            )

    elif teste == "c":
        if gr:
            gr = False
            v = cria_uk(nf, m, t, posic, gr)
            gr = True
        else:
            v = cria_uk(nf, m, t, posic, gr)
        w = []
        for i in range(n + 1):
            a = 0
            for j in range(nf):
                a += intens[j] * v[j][i]
            w.append(a)
        if gr:
            plot(
                w,
                "Vetor U(x) intensidades recuperada",
                "Teste C, Temperatura X Posição, time=1, (N = %d)" % (n),
            )

        x = le_matriz()
        stp = int(2048 / n)
        z = []
        for k in range(0, 2049, stp):
            z.append(x[k])

        ra = random.random()
        ra = ra - 0.5
        ra = ra * 2
        eps = 0.01
        for i in range(len(z)):
            z[i] = z[i] * (1 + ra * eps)

        pos = []  # Quero comparar os gráficos de z e w, vai ser legal!
        for j in range(n + 1):
            pos.append((1 / n) * j)
        plt.plot(pos, z, label="extraído do arquivo")
        plt.plot(pos, w, label="intensidades recuperadas")
        plt.xlabel("position")
        plt.ylabel("temperature")
        plt.title("Vetores U(x) - Temperatura x posição - Tempo = 1")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif teste == "d":
        if gr:
            gr = False
            v = cria_uk(nf, m, t, posic, gr)
            gr = True
        else:
            v = cria_uk(nf, m, t, posic, gr)

        w = []
        for i in range(n + 1):
            a = 0
            for j in range(nf):
                a += intens[j] * v[j][i]
            w.append(a)

        x = le_matriz()
        stp = int(2048 / n)
        z = []
        for k in range(0, 2049, stp):
            z.append(x[k])

        ra = random.random()  # Aplicação de ruído
        ra = ra - 0.5
        ra = ra * 2
        eps = 0.01
        for i in range(len(z)):
            z[i] = z[i] * (1 + ra * eps)

        if gr:
            plot(
                w,
                "Vetor U(x) intensidades recuperada",
                "Teste C, Temperatura X Posição, time=1, (N = %d)" % (n),
            )

        pos = []  # Quero comparar os gráficos de z e w, vai ser legal!
        for j in range(n + 1):
            pos.append((1 / n) * j)
        plt.plot(pos, z, label="extraído do arquivo")
        plt.plot(pos, w, label="intensidades recuperadas")
        plt.xlabel("position")
        plt.ylabel("temperature")
        plt.title("Vetores U(x) - Temperatura x posição - Tempo = 1")
        plt.legend()
        plt.grid(True)
        # plt.savefig('imagem.png')
        plt.show()


def plot(vetor, name, titulo):
    """Função que facilita a impressão de gráficos das soluções, funciona para
    uma única curva por vez.

    Args:
        vetor (list): lista com os valores que serão colocadas como eixo y
        name (str)  : nome d curva ser colocado na legenda
        titulo (str): título do gráfico
    """
    position = []
    for i in range(n + 1):
        position.append((1 / n) * i)

    # plt.scatter(position, vetor, marker='*', c='b')
    plt.plot(position, vetor)

    plt.xlabel("position")
    plt.ylabel("temperature")
    plt.title(titulo)
    plt.legend([name])
    plt.grid(True)
    # plt.savefig('filename.png')
    plt.show()


if __name__ == "__main__":
    main_mmq()  # Invoca a função principal do EP
