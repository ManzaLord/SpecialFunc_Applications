#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros ---
T_0 = 100        # Temperatura en el extremo derecho (x=L)
L = 1.0          # Longitud de la barra en milímetros (mm)
alpha_squared = 0.34 # Coeficiente de difusión térmica al cuadrado (mm^2/s)
# --- Discretización ---
Nx = 100         # Número de puntos espaciales
Nt = 200         # Número de puntos temporales
t_vals = np.linspace(0, 2.0, Nt) # t en segundos
x_vals = np.linspace(0, L, Nx) # x en milímetros

# --- Funciones ---
def obt_k_n(num_terms, length):
    k_n_values = np.zeros(num_terms)
    for n in range(1, num_terms + 1):
        k_n_values[n-1] = (n * np.pi) / length
    return k_n_values

def analytical_solution(x, t, T_L, length, diff_coeff_squared, k_values_array):
    # Término de estado estacionario
    steady_state_term = (x / length)

    # Valores de los parametros para n=1
    k1 = k_values_array[0] # k_values_array[0] es k_1
    coef_n1 = (1 - (2 / np.pi))
    term_n1 = coef_n1 * np.sin(k1 * x) * np.exp(-(k1**2) * diff_coeff_squared * t)

    # Sumatoria para n >= 2
    sum_series = 0.0
    for n_i in range(1, len(k_values_array)):
        n_val = n_i + 1
        kn = k_values_array[n_i] # kn es k_n = n*pi/L

        # El coeficiente para n >= 2
        coef_n = (2 / np.pi) * ((-1)**n_val / n_val)
        sum_series += coef_n * np.sin(kn * x) * np.exp(-(kn**2) * diff_coeff_squared * t)

    # Suma final de los terminos
    U_xt = T_L * (steady_state_term + term_n1 + sum_series)

    return U_xt

num_orden = 5
k_n_values = obt_k_n(num_orden, L)
# Crear una matriz para almacenar los resultados de temperatura
U_vals = np.zeros((Nx, Nt))

print("Calculando la distribución de temperatura usando la fórmula proporcionada...")
for i in range(Nx):
    for j in range(Nt):
        U_vals[i, j] = analytical_solution(x_vals[i], t_vals[j], T_0, L, alpha_squared, k_n_values)
print("Cálculo completado.")

# --- Visualización con contornos ---
plt.figure(figsize=(10, 7))

# Define los niveles de temperatura para los contornos.
levels = np.arange(0, T_0 + 1, 9)

# Colores de la imagen
cmap_to_use = 'YlOrRd'

# Crea el gráfico de contorno relleno
contour_fill = plt.contourf(t_vals, x_vals, U_vals, levels=levels, cmap=cmap_to_use, extend='both')

# Agrega las líneas de contorno.
contour_lines = plt.contour(t_vals, x_vals, U_vals, levels=levels, colors='k', linewidths=0.5)

# Barra de color
cbar = plt.colorbar(contour_fill, ticks=levels, format='%.0f')
cbar.set_label('T (°C)')
plt.xlabel('t (s)')
plt.ylabel('x (mm)')
plt.title('Distribución de Temperatura en 1D')
plt.ylim(0, L)
plt.tight_layout()
plt.savefig('Dist_temp_formula_provided.pdf')
plt.show()

# Calculo del error realtivo al limite para t=2s y cuando tiende a infinito

t_error = 2.0 # segundos

# Encontrar el índice de tiempo más cercano a t_error_calc en t_vals
t_idx_error = np.argmin(np.abs(t_vals - t_error))
print(f"Usando t_vals[{t_idx_error}] = {t_vals[t_idx_error]:.3f} s para el cálculo del error.")

# Obtener la solución u(x, t=2s)
u_t_error = U_vals[:, t_idx_error]

# Calcular la solución en estado estacionario
u_ss_x = T_0 * (x_vals / L)

# Calcular el error relativo
mask_x_gt_0 = x_vals > 0.0

relative_error_vals = np.zeros_like(x_vals)
relative_error_vals[mask_x_gt_0] = np.abs(u_t_error[mask_x_gt_0] - u_ss_x[mask_x_gt_0]) / np.abs(u_ss_x[mask_x_gt_0])

# Visualización del Error Relativo
plt.figure(figsize=(10, 6))
plt.plot(x_vals[mask_x_gt_0], relative_error_vals[mask_x_gt_0], label=f'Error Relativo en t = {t_vals[t_idx_error]:.3f} s')
plt.xlabel('Posición $x$ (mm)')
plt.ylabel('Error Relativo $E_r(x, t)$')
plt.title(f'Error Relativo de la Solución Analítica vs. Estado Estacionario a $t = {t_vals[t_idx_error]:.3f}$ s')
plt.grid(True)
plt.legend()
plt.ylim([0, None]) # El error relativo debe ser no negativo
plt.savefig('Relative_Error_2s.pdf')
plt.show()

# Imprimir el error máximo relativo
max_relative_error = np.max(relative_error_vals[mask_x_gt_0])
print(f"Error relativo máximo en t={t_vals[t_idx_error]:.3f} s: {max_relative_error:.4f}")

