import math
import numpy as np

# Constantes 
PI = math.pi
V0 = 1.0
EPS = 1e-6
KMAX = 100000

def hessienne(X):
    # Matrice hessienne de L par rapport à (r, h, lambda)
    r, h, lam = X 

    d2L_dr2 = PI**2 * (12.0 * r**2 + 2.0 * h**2) - 2.0 * PI / 3.0 * lam * h
    d2L_drdh = 4.0 * PI**2 * r * h - 2.0 * PI / 3.0 * lam * r
    d2L_dh2 = 2.0 * PI**2 * r**2

    d2L_drdlam = -2.0 * PI / 3.0 * r * h
    d2L_dhdlam = -PI / 3.0 * r**2
    d2L_dlam2 = 0.0

    return np.array([
        [d2L_dr2,    d2L_drdh,   d2L_drdlam],
        [d2L_drdh,   d2L_dh2,    d2L_dhdlam],
        [d2L_drdlam, d2L_dhdlam, d2L_dlam2],
    ])

def constraint_gradient(X):
    # c(r, h) = pi/3 * r^2 * h - V0
    # Formule: grad c(r, h) = (2 * pi/3 * r * h, pi/3 * r^2)
    r, h, _ = X 
    return np.array([2 * (PI / 3) * r * h, (PI / 3) * r**2])

def lagrangien(X):
    # Formule: L(X) = pi^2 r^2 (r^2 + h^2) - lam * (pi/3 r^2 h - V0)
    #                 |--------f---------|      |-------c-------|
    r, h, lam = X
    f = PI**2 * r**2 * (r**2 + h**2)
    c = PI * r**2 * h / 3.0 - V0 
    return f - lam * c
    

def gradient(X):
    # Calcule les dérivées partielles de L par rapport à r, h et lambda puis renvoyer le gradient
    # c.f. rapport pour les formules
    r, h, lam = X 
    dL_dr = 4.0 * PI**2 * r**3 + 2.0 * PI**2 * r * h**2 - 2.0 * PI * lam * r * h / 3.0
    dL_dh = r**2 * (2.0 * PI**2 * h - PI * lam / 3.0)
    dL_dlam = -(PI * r**2 * h / 3.0 - V0)
    return np.array([dL_dr, dL_dh, dL_dlam])

def exact_solution():
    # Analytical solution X* = (r*, h*, lambda*) of the problem
    h = (6 * V0 / np.pi) ** (1 / 3)
    return np.array([h / np.sqrt(2), h, 6 * np.pi * h])
