import numpy as np
from sympy import *
import math

from numba import jit, njit

D = 0.5  # Коэффициент миграции
a = 2.0  # Коэффициент рождения новых людей
sourceKoeff = 2.0  # Ёмкость среды
deathKoeff = 1.  # Коэффициент смертность населения

N = 100  # количество точек по оси OX (Площадь занимаемая людьми)
x0 = 0.  # начало отрезка
L = 100.  # конец отрезка
h = (L - x0) / (N - 1)  # шаг по OX

KT = 1000  # количество точек по времени
t0 = 0.  # начальный момент времени
T = 100.  # конечный момент времени
tau = (T - t0) / (KT - 1)  # шаг по времени

sigma = tau * D / h ** 2  # sigma - число Куранта

A = np.zeros((N - 2, N - 2))
d = np.zeros((N - 2))


@njit
def thomasAlgorithm(A, d):  # Tridiagonal matrix algorithm . Или метод прогонки
    n = len(d)
    P = np.zeros(n - 1)
    Q = np.zeros(n - 1)
    x = np.zeros(n)
    P[0] = A[0][1] / -A[0][0]
    Q[0] = -d[0] / -A[0][0]
    for i in range(1, n - 1):  # находим прогоночные коэффициенты
        a = A[i][i - 1]
        b = A[i][i]
        c = A[i][i + 1]

        P[i] = c / (-b - a * P[i - 1])
        Q[i] = (a * Q[i - 1] - d[i]) / (-b - a * P[i - 1])
    x[-1] = (A[n - 1][n - 2] * Q[n - 2] - d[n - 1]) / (-A[n - 1][n - 1] - A[n - 1][n - 2] * P[n - 2])
    for i in range(n - 2, -1, -1):  # Находим неизвекстные
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


def showAllConstant():
    print("Коэффициент миграции (D) = " + str(D) + '\tРождения новых людей (a) = ' + str(a) +
          "\nСмертность населения (σ) = " + str(deathKoeff) + "\tЁмкость среды K = " + str(sourceKoeff))
    print("Количество точек по ОХ (N) = " + str(N) + "\tКоличество точек по времени (KT) = " + str(KT))


@njit
def solutionMatrixStart():  # заполнение матрицы решений краевыми условиями
    x = np.linspace(x0, L, N)
    u = np.zeros((N, KT))
    for i in range(0, N):
        u[i][0] = math.exp(-(i) ** 2)
        u[i][-1] = 0
    return u, x


# @jit(nopython=True)
def createAndSolveMatrix(allSourceFraction):  # заполняем трехдиагональную матрицу Ax=d
    u, x = solutionMatrixStart()
    # print(u)
    for i in range(1, KT):
        A[0, 0] = - 2 * sigma - (a * tau * allSourceFraction(u[1][i - 1], i, u[1][i - 1])) - deathKoeff * tau - 1
        A[0, 1] = sigma
        d[0] = (a * tau + 1) * (-u[1][i - 1]) - sigma * u[0][i]

        for j in range(1, N - 3):
            A[j, j - 1] = sigma
            A[j, j] = - 2 * sigma - (
                    a * tau * allSourceFraction(u[j + 1][i - 1], i, u[j][i - 1])) - deathKoeff * tau - 1
            A[j, j + 1] = sigma
            d[j] = (a * tau + 1) * (-u[j + 1][i - 1])

        A[N - 3, N - 3] = - 2 * sigma - (
                a * tau * allSourceFraction(u[N - 2][i - 1], i, u[N - 2][i - 1])) - deathKoeff * tau - 1
        A[N - 3, N - 4] = sigma
        d[N - 3] = (a * tau + 1) * (-u[N - 2][i - 1]) - sigma * u[N - 1][i]
        # print (A,d)
        u[1:N - 1, i] = thomasAlgorithm(A, d)
        # print(u)
    return u, x
