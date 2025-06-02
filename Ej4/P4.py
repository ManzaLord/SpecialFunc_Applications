import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, jn_zeros

#Define los parametros dados en el inciso del problema
a = 2 
v = 1      
k00 = jn_zeros(0,1)[0]

#Crea las variables independientes y general una malla de coordenadas
r = np.linspace(0, a, 200)
theta = np.linspace(0, 2 * np.pi, 200)
r, theta = np.meshgrid(r, theta)

#Define las coordenadas cartesianas en funcion de las variables independientes
x = r * np.cos(theta)
y = r * np.sin(theta)

#Valores de t elegidos
for t in range(1,4):

    #Define la coordenada z en funcion del tiempo
    z = j0(k00 * r) * np.cos(k00 * v * t)

    #Grafica la membrana
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x,y,z, cmap='winter')
    ax.set_title(f'Membrana del tambor en t = {t}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-1, 1)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(f"Membrana {t}.pdf")
    plt.show()