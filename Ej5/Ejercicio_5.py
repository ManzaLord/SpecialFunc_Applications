#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def calc_elip(a,b):
    m = (a**2 - b**2)/a**2

    I = 4*a*special.ellipe(m)
    return I

def calc_period(theta_0, l, g):
    theta_r = (theta_0*np.pi)/180
    m = (np.sin(theta_r)) / 2

    I = 4*(np.sqrt(l/g))*special.ellipk(m)
    return I

def pend_simp(l, g):
    return 2*np.pi*np.sqrt(l/g)

def relativ_err(T,  Ts):
    return np.abs(T-Ts)/Ts

print("El perimetro de la elipse es:")
print(calc_elip(8,4))

Pend_a = calc_period(70,  0.8, 9.78)
print("El periodo del pendulo para el inciso a, es:")
print(Pend_a)
print("Con un error relativo de: ", relativ_err(Pend_a, pend_simp(0.8, 9.78)))

Pend_b = calc_period(6, 0.8, 9.78)
print("El periodo del pendulo para el inciso b, es:")
print(Pend_b)
print("Con un error relativo de: ", relativ_err(Pend_b, pend_simp(0.8, 9.78)))

