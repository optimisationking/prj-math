import problem as p
import numpy as np
import input as inp
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
    X0, _ = inp.read_input()
    X_star = p.exact_solution()

    print("Méthode 2 : Méthode de gradient à pas optimal")
    print()

    print("Données")
    print("V0   :", p.V0)
    print("X0   :", X0)
    print("eps  :", p.EPS)
    print("KMAX :", p.KMAX)
    print()

    debut = time.perf_counter()
    Xk, k = methode2(X0)
    fin = time.perf_counter()
    duree = fin - debut

    print("Résultats")
    print("Solution pas optimal :", Xk)
    print("Solution exacte      :", X_star)
    print("Itérations           :", k)
    print("||grad L||           :", np.linalg.norm(p.gradient(Xk)))
    print("Erreur               :", np.linalg.norm(Xk - X_star))
    print(f"Durée d'exécution    : {duree:.6f} secondes")

