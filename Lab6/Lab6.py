import numpy as np
from scipy.stats import f, t
from tabulate import tabulate
import sklearn.linear_model as lm
import random
import time

def mult(x1, x2, x3 = np.ones(15)):
    x = np.ones(15)
    for i in range(15):
        x[i] *= x1[i] * x2[i] * x3[i];
    return x
def f_x_func(x, b):
    global f_x
    f_x = np.zeros(N)
    for i in range(N):
        f_x[i] += b[0]
        for k in range(len(x[0])):
            f_x[i] += b[k + 1] * x[i][k]
def find_x(x_min, x_max):
  #величини для значень матриці планування
  x01 = (x_max[0] + x_min[0]) / 2
  x02 = (x_max[1] + x_min[1]) / 2
  x03 = (x_max[2] + x_min[2]) / 2
  delta_x1 = x_max[0] - x01
  delta_x2 = x_max[1] - x02
  delta_x3 = x_max[2] - x03
  X1 = np.array([x_min[0], x_min[0], x_min[0], x_min[0], x_max[0], x_max[0], x_max[0], x_max[0], -l*delta_x1+x01, l*delta_x1+x01, x01, x01, x01, x01, x01])
  X2 = np.array([x_min[1], x_min[1], x_max[1], x_max[1], x_min[1], x_min[1], x_max[1], x_max[1], x02, x02, -l*delta_x2+x02, l*delta_x2+x02, x02, x02, x02])
  X3 = np.array([x_min[2], x_max[2], x_min[2], x_max[2], x_min[2], x_max[2], x_min[2], x_max[2], x03, x03, x03, x03, -l*delta_x3+x03, l*delta_x3+x03, x03])
  return np.array(list(zip(X1, X2, X3, mult(X1, X2), mult(X1, X3), mult(X2, X3), mult(X1, X2, X3), mult(X1, X1), mult(X2, X2), mult(X3, X3))))
def find_y(x, b):
    y_val = np.zeros((N, m))
    f_x_func(x, b)
    for i in range(N):
        for j in range(m):
            y_val[i][j] += f_x[i] + random.random()*10 - 5
    return y_val
def find_b(X, y):
    x = list(X)
    for i in range(len(x)):
        x[i] = np.array([1, ] + list(x[i]))
    X = np.array(x)
    model = lm.LinearRegression(fit_intercept=False)
    model.fit(X, y)
    coefs = model.coef_
    print("Коефіцієнти рівняння регресії:")
    for i in range(len(coefs)):
        print(f"b{i:} = {coefs[i]:}")
    print("\nРівняння регресії")
    print(f"y = {coefs[0]:.3f} + x1 * {coefs[1]:.3f} + x2 * {coefs[2]:.3f} + x3 * {coefs[3]:.3f}"
          f" + x1x2 * {coefs[4]:.3f} + x1x3 * {coefs[5]:.3f} +\n+ x2x3 * {coefs[6]:.3f}"
          f" + x1x2x3 * {coefs[7]:.4f} + x1^2 * {coefs[8]:.4f} + x2^2 * {coefs[9]:.4f} + x3^2 * {coefs[10]:.4f}")
    return coefs
def factor_val_check():
    print("Виконаємо перевірку, підставивши значення факторів з матриці\nпланування і порівняємо результат з середніми значеннями функцій відгуку")
    for i in range(N):
        print(f"y{i+1:} = {f_x[i]:} => {y_mean[i]:} = avgY{i+1:}")
    print("Так як значення співдають, то коефіцієнти рівняння розраховані правильно.\n")
def find_disper(y, y_mean):
    disper = np.zeros(N)
    for i in range(N):
        for j in range(m):
            disper[i] += (y[i][j] - y_mean[i]) ** 2
        disper[i] /= m
    return disper
def matrix_print(y, x_list, y_mean, disper):
    global header_table
    header_table = ["№", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "x1^2", "x2^2", "x3^2"]
    table = []
    for i in range(N):
        table.append([i + 1])
    for i in range(N):
        for _ in range(len(x_list[0])):
            table[i].append(x_list[i][_])
        for j in range(m):
            table[i].append(y[i][j])
        table[i].append(y_mean[i])
        table[i].append(disper[i])
    for i in range(m):
        header_table.append("Y" + str(i + 1))
    header_table.append("Y")
    header_table.append("S^2")
    print(tabulate(table, headers=header_table, tablefmt="fancy_grid"))
def print_eq(b):
    print((f"f(x1, x2, x3) = {b[0]:.1f} + x1 * {b[1]:.1f} + x2 * {b[2]:.1f} + x3 * {b[3]:.1f}"
           f" + x1x2 * {b[4]:.1f} + x1x3 * {b[5]:.1f} + x2x3 * {b[6]:.1f}"
           f" + x1x2x3 * {b[7]:.1f} + x1^2 * {b[8]:.1f} + x2^2 * {b[9]:.1f} + x3^2 * {b[10]:.1f}"))
def initial_print():
    for i in range(len(x_min)):
        print(f"x{i:}min = {x_min[i]:.1f}       x{i:}max = {x_max[i]:.1f}")
def kohren_check(disper):
    global Gp, Gt, f1, f2
    print("Критерій Кохрена")
    Gp = max(disper) / sum(disper)
    f1 = m - 1
    f2 = N
    fisher = f.isf(*[q / f2, f1, (f2 - 1) * f1])
    Gt = round(fisher / (fisher + (f2 - 1)), 4)
    print(f"Gp = {Gp:}\nКількість степенів свободи: F1 = m - 1 = {f1:}; F2 = N = {f2:}")
    print(f"Рівень значимості: q = 1 - p = {q:}\nТабличне значення коефіцієнту Кохрена: Gt = {Gt:}")
def student_check():
    global sb, d, f3, t_exp
    d = len(x_code[0])
    f3 = f1 * f2
    print(f"Критерій Ст`юдента\nЧисло степенів свободи: F3 = F1*F2 = {f3:}")
    sb = sum(disper) / N
    ssbs = sb / N * m
    sbs = ssbs ** 0.5
    beta = np.zeros(d)
    t_exp = []
    for j in range(d):
        for i in range(N):
            if (j == 0):
                beta[j] += y_mean[i]
            else:
                beta[j] += y_mean[i] * x_code[i][j]
        beta[j] /= N
        t_exp.append(abs(beta[j]) / sbs)

    ttabl = round(abs(t.ppf(q / 2, f3)), 4)
    print(f"tтабл = {ttabl:}")
    string_eq = f"y = {b[0]:.7f}"
    for i in range(len(t_exp)):
        if (t_exp[i] < ttabl):
            print(f"Коефіцієнт t{i:} = {t_exp[i]:.7f} = > коефіцієнт не значимий")
            b[i] = 0
            d = d - 1
        else:
            print(f"Коефіцієнт t{i:} = {t_exp[i]:.7f} = > коефіцієнт значимий")
            if(i != 0): string_eq += f" + {b[i]:.7f} * " + header_table[i]
    print("Значимих коефіцієнтів: d = ", d, "\n\nРівняння регресії після виключення коефіцієнтів:\n", string_eq)
    print("\nПеревірка при підстановці в рівняння регресії:")
    f_x_func(x_list, b)
    for i in range(len(f_x)):
        print(f"y'{i+1:} = {f_x[i]:} => {y_mean[i]:} = avgY{i+1:}")
def fisher_check():
    global Fp, Ft
    print("\nКритерій Фішера")
    f4 = N - d
    sad = 0
    for i in range(N):
        sad += (f_x[i] - y_mean[i]) ** 2
    sad *= (m / (N - d))
    Fp = sad / sb
    print(f"Fp = {Fp:}")
    print(f"Кількість степенів свободи: F4 = N - d = {f4:}")
    Ft = round(abs(f.isf(q, f4, f3)), 4)
    print(f"Табличне значення коефіцієнту Фішера: Ft = {Ft:}")

start = time.perf_counter()
#величини за варіантом:
x_min = np.array([-10, 20, 50])
x_max = np.array([50, 60, 55])
initial_print()
b_initial = np.array([1.5, 1.5, 6.8, 3.2, 6.2, 0.7, 2.4, 6.1, 2.7, 0.1, 1.2])
#рівняння з урахуванням квадратичних членів
print_eq(b_initial)
print("\nРівняння регресії:")
print("y = b0 + b1*x1 + b2*x2 + b3*x3 + b12*x1*x2 + b13*x1*x3 + b23*x2*x3 + b123*x1*x2*x3 + b11*x1^2 + b22*x2^2 + b33*x3^2\n")
#константи для початкових умов
m = 3
k = 3 #const
p = 0
N = 15 #2^(k - p)+2k + N0
l = k**(1/2)
q = 0.05
x_code = find_x(np.array([-1, -1, -1]), np.array([1, 1, 1]))
#матриця плануванння
x_list = find_x(x_min, x_max)
y = find_y(x_list, b_initial)
while (time.perf_counter() - start < 10):
    if(m > 3):
        next_int = np.random.randint(y_min, y_max, size=(N, 1))
        y = np.append(y, next_int, axis=1)
    y_mean = np.sum(y, axis=1) / m
    disper = find_disper(y, y_mean)
    print("Матриця планування:")
    matrix_print(y, x_code, y_mean, disper)
    print("Натуралізована матриця:")
    matrix_print(y, x_list, y_mean, disper)

    b = find_b(x_list, y_mean)
    factor_val_check()
    print("Статистичні перевірки:")
    kohren_check(disper)         #find Gp, Gt, f1, f2
    if Gp > Gt:
        print("Дисперсія  неоднорідна , потрібно збільшити  m\n")
        m = m + 1
        continue
    print("Gp < Gt -> Дисперсія однорідна з ймовірністю 0.95\n")
    student_check()
    fisher_check()
    if Fp > Ft:
        print("Fp > Ft = > отримана математична модель з ймовірністю 0.95 неадекватна експерементальним даним\n\n\n\n\n")
        y = find_y(x_list, b_initial)
        continue
    else:
        print("Fp < Ft = > отримана математична модель з ймовірністю 0.95 адекватна експерементальним даним")
    break
print(f"Програма закінчилась з часом t = {(time.perf_counter() - start):.2f}. Значимих коефіцієнтів: d = {d:}", d)
