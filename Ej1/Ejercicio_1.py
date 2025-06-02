#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# Valores de x para intervalo x>0
x_0 = np.linspace(0, 7, 100)

# Valores de x para intervalo -4<x<4
x_1 = np.linspace(-4, 4, 100)

# Valores de x para intervalo mas grande positivo
x_2 = np.linspace(0, 15, 100)

# Valores de x para intervalo -8<x<8
x_3 = np.linspace(0, 10, 100)

# Polinomios de Laguerre
L0 = special.eval_laguerre(0, x_0)
L1 = special.eval_laguerre(1, x_0)
L2 = special.eval_laguerre(2, x_0)
L3 = special.eval_laguerre(3, x_0)

# Graficacion de polinomios Laguerre
plt.figure(1, figsize=(10,8))
plt.plot(x_0, L0, label=f'$L_0(x)$')
plt.plot(x_0, L1, label=f'$L_1(x)$')
plt.plot(x_0, L2, label=f'$L_2(x)$')
plt.plot(x_0, L3, label=f'$L_3(x)$')
plt.xlim(0, 7)
plt.ylim(-4, 4)
plt.xlabel('x')
plt.ylabel('$L_n(x)$')
plt.grid(True)
plt.legend()
plt.savefig('Pol_Laguerre.pdf')
plt.show()

# Polinomios de Hermite
H0 = special.eval_hermite(0, x_1)
H1 = special.eval_hermite(1, x_1)
H2 = special.eval_hermite(2, x_1)
H3 = special.eval_hermite(3, x_1)

# Graficacion porlinomios de Hermite
plt.figure(2, figsize=(10,8))
plt.plot(x_1, H0, label=f'$H_0(x)$')
plt.plot(x_1, H1, label=f'$H_1(x)$')
plt.plot(x_1, H2, label=f'$H_2(x)$')
plt.plot(x_1, H3, label=f'$H_3(x)$')
plt.xlim(-4, 4)
plt.ylim(-10, 10)
plt.xlabel('x')
plt.ylabel('$H_n(x)$')
plt.grid(True)
plt.legend()
plt.savefig('Pol_Hermite.pdf')
plt.show()

# Funciones de Bessel con indice entero
J0 = special.jv(0, x_2)
J1 = special.jv(1, x_2)
J2 = special.jv(2, x_2)
J3 = special.jv(3, x_2)

# Graficacion de las funciones de Bessel indice entero
plt.figure(3, figsize=(10,8))
plt.plot(x_2, J0, label=f'$J_0(x)$')
plt.plot(x_2, J1, label=f'$J_1(x)$')
plt.plot(x_2, J2, label=f'$J_2(x)$')
plt.plot(x_2, J3, label=f'$J_3(x)$')
plt.xlim(0, 15)
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('$J_n(x)$')
plt.grid(True)
plt.legend()
plt.savefig('Func_Bessel_ent.pdf')
plt.show()

# Funciones de Bessel con indice semi-entero
Jm12 = special.jv(-1/2, x_2)
J12 = special.jv(1/2, x_2)
J32 = special.jv(3/2, x_2)

# Graficacion de las funciones de Bessel con indice semi-entero
plt.figure(4, figsize=(10,8))
plt.plot(x_2, Jm12, label=r'$J_{-1/2}(x)$')
plt.plot(x_2, J12, label=r'$J_{1/2}(x)$')
plt.plot(x_2, J32, label=r'$J_{3/2}(x)$')
plt.xlim(0, 15)
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('$J_n(x)$')
plt.grid(True)
plt.legend()
plt.savefig('Func_Bessel_semient.pdf')
plt.show()

# Funcion de Bessel de segunda clase con primera clase semi-entero
Y12 = special.yv(1/2, x_2)

# Graficacion de la funcion de Bessel de segunda clase con primera clase semi-entero
plt.figure(4, figsize=(10,8))
plt.plot(x_2, Y12, label=r'$Y_{-1/2}(x)$')
plt.plot(x_2, J12, label=r'$J_{1/2}(x)$')
plt.xlim(0, 15)
plt.ylim(-1.5, 1)
plt.xlabel('x')
plt.ylabel('$Y/J_n(x)$')
plt.grid(True)
plt.legend()
plt.savefig('Func_BesselIIclas_semient.pdf')
plt.show()

# Funciones de Hankel esfericas
j1 = special.spherical_jn(1, x_3)  # Esferica de primera clase
j2 = special.spherical_jn(2, x_3)
y1 = special.spherical_yn(1, x_3)  # Esferica segunda clase
y2 = special.spherical_yn(2, x_3)

print(special.spherical_jn(1, -4))

h11 = j1 + 1j*y1  # Primera clase n=1
h21 = j1 - 1j*y1  # Segunda clase n=1
h12 = j2 + 1j*y2  # Primera clase n=2
h22 = j2 - 1j*y2  # Segunda clase n=2

# Graficacion de las funciones de Hankel esfericas
plt.figure(5, figsize=(10,8))
plt.plot(x_3, j1, label=f'Re($h^1_1(x)$)')
plt.plot(x_3, h11.imag, linestyle='--', label=f'Im($h^1_1(x)$)')

plt.plot(x_3, j1, label=f'Re($h^2_1(x)$)')
plt.plot(x_3, h21.imag, linestyle='--', label=f'Im($h^2_1(x)$)')

plt.plot(x_3, j2, label=f'Re($h^1_2(x)$)')
plt.plot(x_3, h12.imag, linestyle='--', label=f'Im($h^1_2(x)$)')

plt.plot(x_3, j2, label=f'Re($h^2_2(x)$)')
plt.plot(x_3, h22.imag, linestyle='--', label=f'Im($h^2_2(x)$)')

plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('$h_{n}^{(1,2)}(x)$')
plt.grid(True)
plt.legend()
plt.savefig('Func_BesselIIIesfer.pdf')
plt.show()

