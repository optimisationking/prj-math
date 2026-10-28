import problem as p
import numpy as np
import time

#Methode 2 : Gradient à pas optimale

def line_search(X0, d, alpha_init=1.0, Beta=0.5, c=1e-4):
    #d the direction of descent
    #alpha_init we define an initial step that is overly large to then reduce it
    #Beta the amount we reduce our step
    #c the constant to define the level of acceptance of the search
    
    
    alpha = alpha_init
    grad = p.gradient(X0)

    #till the armijo condition is fulfilled, we decrease the step size
    while p.lagrangien(X0 + alpha*d) > (p.lagrangien(X0) + c * alpha * np.dot(d, grad)):
        alpha *= Beta
        if alpha < 1e-12:       #if alpha gets too small, break the cycle
            break
    return alpha


def methode2(X0, max_iter=10000):
    k = 0
    Xk = np.array(X0, dtype=float).copy()
    norm = 1.0
    while norm > p.EPS and k < max_iter:
        dk = -p.gradient(Xk)
        alphaK = line_search(Xk, dk)
        Xk += alphaK*dk
        norm = np.linalg.norm(p.gradient(Xk))
        k += 1
    return Xk, k

if __name__ == "__main__":
    debut = time.perf_counter()
    print(methode2([1,1,1], 0.01))
    fin = time.perf_counter()
    duree = fin - debut
    print(f"Durée d'exécution : {duree:.6f} secondes")


