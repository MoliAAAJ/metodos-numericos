import math
import numpy as np
import matplotlib
matplotlib.use('QtAgg') # Esto le dice que use Qt (detectará Qt6 automáticamente) - KDE Plasma
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Ejemplo de error numérico

a = 0.1 + 0.2
print("error numérico:", a)

# Ejemplo de error absoluto y relativo

real = math.pi
aprox = 3.14

error_abs = abs(real - aprox)
error_rel = abs(real - aprox) / real

print("error absoluto:", error_abs)
print("error relativo:", error_rel)

# Función Bisección para encontrar raíces

def biseccion(f, a, b, tol=1e-6, max_iter=100): # f es la función, a y b son los extremos del intervalo, tol es la tolerancia y max_iter es el número máximo de iteraciones

    if f(a) * f(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")

    for i in range(max_iter):

        c = (a + b) / 2

        if abs(f(c)) < tol:
            return c

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2

# Ejemplo de uso de la función bisección

def f(x):
    return x**3 - x - 2

raiz = biseccion(f, 1, 2)

print("raiz aproximada:", raiz)

# Visualización de la función

x = np.linspace(0, 3, 100)
y = x**3 - x - 2

plt.axhline(0)
plt.plot(x, y)
plt.show()

# Método de Newton-Raphson para encontrar raíces

def newton(f, df, x0, tol=1e-6, max_iter=100): # f es la función, df es su derivada, x0 es la aproximación inicial, tol es la tolerancia y max_iter es el número máximo de iteraciones

    x = x0

    for i in range(max_iter):

        x_new = x - f(x)/df(x)

        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    return x

# Ejemplo de uso del método de Newton-Raphson

def f(x):
    return x**2 - 2

def df(x):
    return 2*x

raiz = newton(f, df, 1)

print("raizNewton:", raiz)

# Visualización de la función

x = np.linspace(0, 2, 100)
y = x**2 - 2

plt.axhline(0)
plt.plot(x, y)
plt.show()

# Método de la secante para encontrar raíces

def secante(f, x0, x1, tol=1e-6, max_iter=100): # f es la función, x0 y x1 son las aproximaciones iniciales, tol es la tolerancia y max_iter es el número máximo de iteraciones

    for i in range(max_iter):

        x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))

        if abs(x2 - x1) < tol:
            return x2

        x0 = x1
        x1 = x2

    return x1

# Ejemplo de uso del método de la secante

def f(x):
    return x**3 - x - 2

raiz = secante(f, 1, 2)

print("raizSec:", raiz)

# Visualización de la función

x = np.linspace(0, 3, 100)
y = x**3 - x - 2

plt.axhline(0)
plt.plot(x, y)
plt.show()

# Interpolación de Lagrange

def lagrange(x, y, xp): # x es el vector de los puntos de interpolación, y es el vector de los valores de la función en esos puntos, xp es el punto donde se quiere evaluar la interpolación

    n = len(x)
    yp = 0

    for i in range(n):

        Li = 1

        for j in range(n):

            if i != j:
                Li *= (xp - x[j])/(x[i] - x[j])

        yp += y[i]*Li

    return yp

# Ejemplo de uso de la interpolación de Lagrange

# Datos
x = [0,1,2]
y = [1,3,2]

# Interpolar en
xp = 1.5

resultado = lagrange(x,y,1.5)

print("Lagrange:", resultado)

# Visualización de los puntos y la interpolación

xp = np.linspace(0,2,100)
yp = [lagrange(x,y,val) for val in xp]

plt.scatter(x,y)
plt.plot(xp,yp)

plt.show()

# Splines cúbicos (interpolación por partes con polinomios cúbicos)

x = [0,1,2,3]
y = [1,3,2,5]

spline = CubicSpline(x,y) # Crea un objeto spline a partir de los puntos (x,y)

xp = np.linspace(0,3,100)
yp = spline(xp)

plt.scatter(x,y)
plt.plot(xp,yp)

plt.show()

# Integración numérica con el método del trapecio

def trapecio(f, a, b, n): # f es la función a integrar, a y b son los límites de integración, n es el número de subintervalos

    h = (b - a) / n
    suma = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        suma += f(a + i*h)

    return h * suma

# Ejemplo de uso del método del trapecio

def f(x):
    return x**2

resultado = trapecio(f,0,1,100)

print("Trapecio:", resultado)

# Visualización de la función y el área bajo la curva

x = np.linspace(0,1,100)
y = x**2

plt.plot(x,y)
plt.fill_between(x,y,alpha=0.5)
plt.show()

# Regla de Simpson para integración numérica

def simpson(f, a, b, n): # f es la función a integrar, a y b son los límites de integración, n es el número de subintervalos (debe ser par)

    if n % 2 != 0:
        raise ValueError("n debe ser par")

    h = (b - a) / n
    suma = f(a) + f(b)

    for i in range(1, n):

        x = a + i*h

        if i % 2 == 0:
            suma += 2*f(x)
        else:
            suma += 4*f(x)

    return h/3 * suma

# Ejemplo de uso de la regla de Simpson

def f(x):
    return math.sin(x)

resultado = simpson(f,0,math.pi,100)
print("Simpson:", resultado)

# Visualización de la función y el área bajo la curva

x = np.linspace(0,math.pi,100)
y = np.sin(x)

plt.plot(x,y)
plt.fill_between(x,y,alpha=0.5)
plt.show()

# Derivación numérica (Derivada central)

def derivada_central(f, x, h=1e-5): # f es la función, x es el punto donde se quiere evaluar la derivada, h es el paso (tamaño del intervalo)

    return (f(x+h) - f(x-h)) / (2*h)

# Ejemplo de uso de la derivada central

def f(x):
    return x**2

resultado = derivada_central(f,2)

print("Derivada central:", resultado)

# Visualización de la función y su derivada

x = np.linspace(0,4,100)
y = x**2
dy = 2*x

plt.plot(x,y,label='f(x)')
plt.plot(x,dy,label="f'(x)")
plt.legend()
plt.show()

# Sistemas de ecuaciones lineales (Método de Gauss)

A = np.array([[2,1],
              [1,3]])

b = np.array([5,6])

x = np.linalg.solve(A,b) # Resuelve el sistema de ecuaciones Ax = b utilizando el método de Gauss (o cualquier método de resolución de sistemas lineales)

print("Solución del sistema:", x)

# Verificación de la solución

print("Verificación:", A @ x)

# Visualización de las ecuaciones

x1 = np.linspace(0,5,100)
y1 = (5 - 2*x1) / 1 # De la primera ecuación: 2x + y = 5 => y = (5 - 2x)
y2 = (6 - 1*x1) / 3 # De la segunda ecuación: 1x + 3y = 6 => y = (6 - 1x) / 3

plt.plot(x1,y1,label='2x + y = 5')
plt.plot(x1,y2,label='x + 3y = 6')
plt.legend()
plt.show()

# Métodos Iterativos para sistemas de ecuaciones lineales (Jacobi)

def jacobi(A, b, x0, tol=1e-6, max_iter=100): # A es la matriz de coeficientes, b es el vector de términos independientes, x0 es la aproximación inicial, tol es la tolerancia y max_iter es el número máximo de iteraciones

    n = len(b)
    x = x0.copy()

    for _ in range(max_iter):

        x_new = np.zeros_like(x)

        for i in range(n):

            s = 0

            for j in range(n):
                if j != i:
                    s += A[i][j]*x[j]

            x_new[i] = (b[i] - s)/A[i][i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new

    return x

# Ejemplo de uso del método de Jacobi

A = np.array([[4,1],
              [1,3]])

b = np.array([9,7])

x0 = np.array([0.0,0.0])

sol = jacobi(A,b,x0)

print("Jacobi: ", sol)

# Ecuaciones diferenciales Ordinarias (Método de Euler)

def euler(f, x0, y0, h, n): # f es la función que define la ecuación diferencial (dy/dx = f(x,y)), x0 y y0 son las condiciones iniciales, h es el paso, n es el número de iteraciones

    x = x0
    y = y0

    xs = [x]
    ys = [y]

    for i in range(n):

        y = y + h*f(x,y)
        x = x + h

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Ejemplo de uso del método de Euler

def f(x,y):
    return y

x,y = euler(f,0,1,0.1,20)

print("Euler: ", y)

# Visualización de la solución

import matplotlib.pyplot as plt

plt.plot(x,y)
plt.show()

# Método de Runge-Kutta de orden 4 para ecuaciones diferenciales ordinarias

def rk4(f, x0, y0, h, n):

    x = x0
    y = y0

    xs = [x]
    ys = [y]

    for i in range(n):

        k1 = f(x,y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)

        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x = x + h

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Ejemplo de uso del método de Runge-Kutta de orden 4

def f(x,y):
    return y

x,y = rk4(f,0,1,0.1,20)

print("Runge-Kutta 4: ", y)

# Visualización de la solución

plt.plot(x,y)
plt.show()
