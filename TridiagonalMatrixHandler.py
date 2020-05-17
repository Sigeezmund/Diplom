import numpy as np
import math

from numba import njit

D = 0.5  # Коэффициент миграции
birthKoeff = 2.0  # Коэффициент рождения новых людей
deathKoeff = 1.  # Коэффициент смертность населения

NX = 500  # количество точек по оси OX (Площадь занимаемая людьми)
x0 = 0  # начало отрезка
L = 1000  # конец отрезка
h = (L - x0) / (NX - 1)  # шаг по OX

KT = 1000  # количество точек по времени
t0 = 0  # начальный момент времени
T = 1000  # конечный момент времени
tau = (T - t0) / (KT - 1)  # шаг по времени

sigma = tau * D / h ** 2  # sigma - число Куранта

x = np.linspace(x0, L, NX)


def showAllConstant():  # Метод для отображение заданных коэффициентов.
    print("Коэффициент миграции (D) = " + str(D) + '\tРождения новых людей (a) = ' + str(birthKoeff) +
          "\tСмертность населения (σ) = " + str(deathKoeff))
    print()
    print("Количество точек по ОХ (N) = " + str(NX) + "\t\tКоличество точек по времени (KT) = " + str(KT))
    print("Конец отрезка по OX (L) = " + str(L) + "\t\tКонечный момент (T) = " + str(T))
    print()
    print("число Куранта = " + str(sigma) + str('\tСистема устойчивая' if sigma <= 0.5 else '\tСистема не устойчива'))


# Заполнение начальной матрицы нулями и начальными условиями в границах,
# и также обозначение начального рассспределения
@njit
def getStartMatrix():
    u = np.zeros((NX, KT), dtype=np.float64)
    for i in range(1, NX):
        u[i][0] = math.exp(-(i) ** 2)
    for i in range(0, KT):
        u[0][i] = 2
    return u


# высчитывание интеграла как плошадь под криволинейной трапецей. Сделанно потому что традиционные способы интегрирования
# не адаптированны под нумбу
@njit
def numbaQuad(u):
    inter = 0
    for j in range(NX):
        inter = inter + u[j]
    inter *= h
    return inter


# Решение с помощью явной схемы
@njit
def createAndSolveUByYavnayMethods(carryingCapacityFunction):
    u = getStartMatrix()
    u_0 = np.zeros(NX)
    for k in range(0, KT - 1):
        for j in range(NX):
            ujk = u[j][k]
            if j == 0:
                u[j][k + 1] = sigma * (u[j + 1][k] - 2 * ujk) + tau * birthKoeff * ujk * (
                        1 - carryingCapacityFunction(ujk, k, u_0)) - tau * deathKoeff * ujk + ujk
            if j == NX - 1:
                u[j][k + 1] = sigma * (-2 * ujk + u[j - 1][k]) + tau * birthKoeff * ujk * (
                        1 - carryingCapacityFunction(ujk, k, u_0)) - tau * deathKoeff * ujk + ujk
            else:
                u[j][k + 1] = sigma * (u[j + 1][k] - 2 * ujk + u[j - 1][k]) + tau * birthKoeff * ujk * (
                        1 - carryingCapacityFunction(ujk, k, u_0)) - tau * deathKoeff * ujk + ujk
            u_0 = u[0:NX, k]
    return u


# Tridiagonal matrix algorithm . Или метод прогонки
@njit
def thomasAlgorithm(A, d):
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


# заполняем трехдиагональную матрицу Ax=d. Или решение не явной схемой
@njit
def createAndSolveUNeYavnayaMethods(carryingCapacityFunction):
    A = np.zeros((NX - 2, NX - 2))
    d = np.zeros((NX - 2))
    u = getStartMatrix()
    a = sigma
    b = lambda ukj: -2 * sigma - (birthKoeff * tau * ukj) / carryingCapacityFunction() - 1 - tau * deathKoeff
    c = sigma
    for k in range(1, KT):
        for j in range(0, NX - 3):
            if j == 0:
                A[j][j] = b(u[j + 1][k - 1])
                A[j][j + 1] = c
                d[j] = -(birthKoeff * tau + 1) * u[j + 1][k - 1] - sigma * u[j][k - 1]
            if j == NX - 3:
                A[j][j + 1] = a
                A[j][j] = b(u[j][k - 1])
                d[j] = -(birthKoeff * tau + 1) * u[j][k - 1] - sigma * u[j + 1][k - 1]
            else:
                A[j][j - 1] = a
                A[j][j] = b(u[j][k - 1])
                A[j][j + 1] = c
                d[j] = -(birthKoeff * tau + 1) * u[j][k - 1]
        u[1:NX - 1, k] = thomasAlgorithm(A, d)
        # print(u)
    return u
