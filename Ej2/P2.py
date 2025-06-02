
import matplotlib.pyplot as plt
import numpy as np

#Define un polinomio de Legendre de grado l
def polLegendre(grado):
    return np.polynomial.legendre.Legendre.basis(grado)

#Define un polinomio de Chebyshev de grado m
def polChebyshev(grado):
    return np.polynomial.chebyshev.Chebyshev.basis(grado)

#Define la funcion peso del polinomio de Chebyshev
def weightFunc(x):
    return 1 / np.sqrt(1 - x**2)

def plotFunc(x, data, name):
    #Grafica la senal
    plt.figure(figsize=(10,5))
    plt.plot(x,data)
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.title(name)
    
    #Guarda la grafica
    plt.savefig(f"{name}.pdf")
    plt.grid(True)
    plt.show()

#Define el intervalo de los polinomios
x = np.linspace(-1,1,1000)

#Calcula los polinomios
pL = polLegendre(7)
pC = polChebyshev(8)

#Acomoda el integrando de la funcion sin evaluar x
producto = pL(x) * pC(x) * weightFunc(x)

#Grafica la funcion Legendre - Chebyshev
plotFunc(x,producto,'Producto Legendre - Chebyshev')

#Calcula los polinomios
pL1 = polChebyshev(2)
pL2 = polChebyshev(6)

#Acomoda el integrando de la funcion sin evaluar x
producto = pL1(x) * pL2(x) * weightFunc(x)

#Grafica la funcion Chebyshev - Chebyshev
plotFunc(x,producto,'Producto Chebyshev - Chebyshev')