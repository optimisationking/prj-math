import problem as p
import numpy as np
import input as inp
import time

# Méthode 1 : Gradient à pas fixe 

def direction(gradk):
    dk = -gradk 
    dk[2] = -dk[2] # dual correction
    return dk

def updateX(Xk, alpha, dk):
    return Xk + alpha * dk

def methode1(X0, alpha):
    Xk = np.array(X0, dtype=float) 
    gradk = p.gradient(Xk)          # grad L(X_0)
    norme = np.linalg.norm(gradk)   # norme de grad L(X_0)
    k = 0

    while norme > p.EPS and k < p.KMAX:
        dk = direction(gradk)           
        Xk = updateX(Xk, alpha, dk)     
        gradk = p.gradient(Xk)          # grad L(X_k+1)
        norme = np.linalg.norm(gradk)   # norme de grad L(X_k+1)
        k += 1
        
        if not np.isfinite(norme):       # the iterates diverged: stop
            break
    
    return Xk, k

if __name__ == "__main__":
    X0, alpha = inp.read_input()
    X_star = p.exact_solution()

    print("Méthode 1 : Méthode de gradient à pas fixe")
    print()

    print("Données")
    print("V0       :", p.V0)
    print("X0       :", X0)
    print("alpha    :", alpha)
    print("eps      :", p.EPS)
    print("KMAX     :", p.KMAX)
    print()

    debut = time.perf_counter()
    Xk, k = methode1(X0, alpha)
    fin = time.perf_counter()
    duree = fin - debut

    print("Résultats")
    print("Solution pas fixe    :", Xk)
    print("Solution exacte      :", X_star)
    print("Itérations           :", k)
    print("||grad L||           :", np.linalg.norm(p.gradient(Xk)))
    print("Erreur               :", np.linalg.norm(Xk - X_star))
    print(f"Durée d'exécution    : {duree:.6f} secondes")
