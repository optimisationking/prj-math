import problem as p
import numpy as np
import time

# Methode 2 : Gradient à pas optimal (règle d'Armijo)

RHO = 1.0           # Fixed step for dual correction
ALPHA_INIT = 1.0    # Initial step that is overly large
BETA = 0.5          # The amount we reduce our step
C1 = 1e-4           # Constant to define the level of acceptance of the search
ALPHA_MIN = 1e-12   # Below this value, the search is stopped

def direction(gradk):
    dk = -gradk 
    dk[2] = 0.0
    return dk

def updateX(Xk, ak, dk, gradk):
    res = Xk + ak * dk 
    res[2] += RHO * gradk[2]        # dual correction
    return res

def armijo_condition(Xk, dk, alpha, Lk, slope):
    return p.lagrangien(Xk + alpha * dk) <= Lk + C1 * alpha * slope

def line_search(Xk, dk, gradk):
    Lk = p.lagrangien(Xk)
    slope = np.dot(gradk, dk)
    alpha = ALPHA_INIT 

    # till the armijo condition is fulfilled, we decrease the step size
    while not armijo_condition(Xk, dk, alpha, Lk, slope):
        alpha *= BETA
        if alpha < ALPHA_MIN:       #if alpha gets too small, break the cycle
            break

    return alpha


def methode2(X0):
    Xk = np.array(X0, dtype=float)
    gradk = p.gradient(Xk)
    norm = np.linalg.norm(gradk)
    k = 0

    while norm > p.EPS and k < p.KMAX:
        dk = direction(gradk)
        ak = line_search(Xk, dk, gradk)
        Xk = updateX(Xk, ak, dk, gradk)       
        gradk = p.gradient(Xk) 
        norm = np.linalg.norm(gradk)
        k += 1
        if not np.isfinite(norm):       # the iterates diverged: stop
            break

    return Xk, k

if __name__ == "__main__":
    debut = time.perf_counter()
    print(methode2([1,1,1]))
    fin = time.perf_counter()
    duree = fin - debut
    print(f"Durée d'exécution : {duree:.6f} secondes")


