import problem as p
import numpy as np
import input as inp
import log as lg
import time

# Methode 2 : Gradient à pas optimal (règle d'Armijo)

LOG_FILE = "resultat_methode2.csv"
LOG_COLUMNS = ("alpha", "n_trials", "armijo_ok")  

ALPHA_INIT = 1.0    # initial step that is overly large
BETA = 0.5          # ihe amount we reduce our step
C1 = 1e-4           # constant to define the level of acceptance of the search
ALPHA_MIN = 1e-12   # below this value, the armijo search is stopped

def direction(gradk):
    return -gradk 

def updateX(Xk, ak, dk):
    return Xk + ak * dk

def armijo_condition(Xk, dk, alpha, Lk, slope):
    return p.lagrangien(Xk + alpha * dk) <= Lk + C1 * alpha * slope

def line_search(Xk, dk, gradk):
    Lk = p.lagrangien(Xk)
    slope = np.dot(gradk, dk)
    alpha = ALPHA_INIT 
    n_trials = 1

    # until the armijo condition is fulfilled, we decrease the step size
    while not armijo_condition(Xk, dk, alpha, Lk, slope):
        alpha *= BETA
        n_trials += 1
        if alpha < ALPHA_MIN:       # if alpha gets too small, break the cycle and return False since condition not fulfilled
            return alpha, n_trials, False 

    return alpha, n_trials, True

def methode2(X0):
    X_star = p.exact_solution()
    Xk = np.array(X0, dtype=float)
    gradk = p.gradient(Xk)
    norm = np.linalg.norm(gradk)
    k = 0
    rows = [lg.make_row(k, Xk, norm, X_star, (np.nan, 0, True))]

    while norm > p.EPS and k < p.KMAX:
        dk = direction(gradk)
        ak, n_trials, armijo_ok = line_search(Xk, dk, gradk)    # get step alpha using armijo and collect other info
        Xk = updateX(Xk, ak, dk)
        gradk = p.gradient(Xk) 
        norm = np.linalg.norm(gradk)
        k += 1
        rows.append(lg.make_row(k, Xk, norm, X_star, (ak, n_trials, armijo_ok)))    # update log

        if not np.isfinite(norm) or np.linalg.norm(Xk) > p.XMAX:       # the iterates diverged: stop
            break

    return Xk, k, rows

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
    Xk, k, rows = methode2(X0)
    fin = time.perf_counter()
    duree = fin - debut

    lg.write_log(LOG_FILE, rows, LOG_COLUMNS) 

    print("Résultats")
    print("Solution pas optimal :", Xk)
    print("Solution exacte      :", X_star)
    print("Itérations           :", k)
    print("||grad L||           :", np.linalg.norm(p.gradient(Xk)))
    print("Erreur               :", np.linalg.norm(Xk - X_star))
    print(f"Durée d'exécution    : {duree:.6f} secondes")

