import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

### однопараметрический анализ
# параметры
k1_val = 0.12
k_1_val = [0.001, 0.005, 0.01, 0.015, 0.02]
k2_val = 0.95
k3_val = 0.0032
k_3_val = [0.0005, 0.001, 0.002, 0.003, 0.004]
k_1_vall = k_1_val[3]
k_3_vall = k_3_val[3]
X = np.linspace(0.001, 0.999, 999)
y0 = [0.2, 0.2]

# уравнение
x, y, k1, k_1, k2, k3, k_3 = sp.symbols('x y k1 k_1 k2 k3 k_3')
z = 1 - x - 2 * y
dx = k1 * z - k_1 * x - k3 * x * z + k_3 * y - k2 * (z ** 2) * x
dy = k3 * x * z - k_3 * y

result = sp.solve([dx, dy], (y, k1))
y_solution, k1_solution = result[0][0], result[0][1]
print(y_solution)
print(k1_solution)

y_function = sp.lambdify((x, k3, k_3), y_solution)
k1_function = sp.lambdify((x, k2, k3, k_1, k_3), k1_solution)

# Матрица Якоби
jacA = sp.Matrix([dx, dy]).jacobian([x, y])
traceA = jacA.trace()
detA = jacA.det()

detA_func = sp.lambdify((x, k1, k2, k3, k_1, k_3), detA.subs(y, y_solution))
traceA_func = sp.lambdify((x, k1, k2, k3, k_1, k_3), traceA.subs(y, y_solution))

K1 = k1_function(X, k2_val, k3_val, k_1_vall, k_3_vall)
Y = y_function(X, k3_val, k_3_vall)

detA_values = detA_func(X, k1_val, k2_val, k3_val, k_1_vall, k_3_vall)
traceA_values = traceA_func(X, k1_val, k2_val, k3_val, k_1_vall, k_3_vall)


def find_zeros(values):
    zeros = []
    for i in range(values.shape[0] - 1):
        if values[i] == 0 or values[i] * values[i + 1] < 0:
            zeros.append(i)
    return np.asarray(zeros)


detA_zeros = find_zeros(detA_values)
traceA_zeros = find_zeros(traceA_values)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(X, detA_values)
plt.title('det')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(X, traceA_values)
plt.title('trace')
plt.grid()

plt.figure()
plt.plot(K1, X, 'k-', label='$x(k)$')
plt.plot(K1, Y, 'b--', label='$y(k)$')

for zero in traceA_zeros:
    plt.plot(K1[zero], X[zero], 'bs', label='hopf')
    plt.plot(K1[zero], Y[zero], 'bs', label='hopf')

for zero in detA_zeros:
    plt.plot(K1[zero], X[zero], 'k*', label='saddle')
    plt.plot(K1[zero], Y[zero], 'k*', label='saddle')

plt.title(f'$k_{-1}$ = {k_1_vall}, $k_{-3}$ = {k_3_vall}')
plt.grid()
plt.xlim(0, 1)
plt.legend()
plt.show()

### Решение системы
k1_val = 0.12
k_1_val = 0.01
k2_val = 0.95
k3_val = 0.0032
k_3_val = 0.002


def func(y, time):
    z = 1 - y[0] - 2 * y[1]
    dx = k1_val * z - k_1_val * y[0] - k3_val * y[0] * z + k_3_val * y[1] - k2_val * (z ** 2) * y[0]
    dy = k3_val * y[0] * z - k_3_val * y[1]
    return [dx, dy]


# point = detA_zeros[0]
t = np.linspace(0, 4000, 10000)
ans = odeint(func, y0, t)

plt.figure()
plt.plot(t, ans[:, 0], color='k', label='$x(t)$')
plt.plot(t, ans[:, 1], color='b', label='$y(t)$')
plt.title('Решение системы')
plt.xlabel('$t$')
plt.grid()
plt.legend()

plt.figure()
plt.plot(ans[:, 0], ans[:, 1], color='r', label='$y(x)$')
plt.title('Фазовый портрет системы')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()
plt.show()

### Двухпараметрический анализ
k2_solution_det = sp.solve(detA.subs({y: y_solution, k1: k1_solution}), k2)[0]
k2_function_det = sp.lambdify((x, k_1, k3, k_3), k2_solution_det)
k2_solution_trace = sp.solve(traceA.subs({y: y_solution, k1: k1_solution}), k2)[0]
k2_function_trace = sp.lambdify((x, k_1, k3, k_3), k2_solution_trace)

k1_solution_det = k1_solution.subs({k2: k2_solution_det})
k1_function_det = sp.lambdify((x, k_1, k3, k_3), k1_solution_det)
k1_solution_trace = k1_solution.subs({k2: k2_solution_trace})
k1_function_trace = sp.lambdify((x, k_1, k3, k_3), k1_solution_trace)

k1_val_det = []
k2_val_det = []
k1_val_trace = []
k2_val_trace = []

for i in range(len(X)):
    k1_val_det.append(k1_function_det(X[i], k_1_val, k3_val, k_3_val))
    k2_val_det.append(k2_function_det(X[i], k_1_val, k3_val, k_3_val))
    k1_val_trace.append(k1_function_trace(X[i], k_1_val, k3_val, k_3_val))
    k2_val_trace.append(k2_function_trace(X[i], k_1_val, k3_val, k_3_val))

plt.plot(k1_val_det, k2_val_det, 'k:', label='кратность')
plt.plot(k1_val_trace, k2_val_trace, 'b--', label='нейтральность')
plt.title('Параметрический портрет')
plt.xlabel('$k_1$')
plt.ylabel('$k_2$')
plt.grid()
plt.legend()
plt.show()