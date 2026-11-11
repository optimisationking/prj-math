import problem as p
import numpy as np
import time

# Méthode 1 : Gradient à pas fixe 

def methode1(X0, alpha):
    Xk = np.array(X0, dtype=float) 
    gradk = p.gradient(Xk)          # grad L(X_0)
    norme = np.linalg.norm(gradk)   # norme de grad L(X_0)
    k = 0

    while norme > p.EPS and k < p.KMAX:
        dk = -gradk                     # direction de descente
        dk[2] = -dk[2]                  # dual correction
        Xk += alpha * dk                # pas fixe
        gradk = p.gradient(Xk)          # grad L(X_k+1)
        norme = np.linalg.norm(gradk)   # norme de grad L(X_k+1)
        k += 1
        
        if not np.isfinite(norme):       # the iterates diverged: stop
            break
    
    return Xk, k

if __name__ == "__main__":
    debut = time.perf_counter()
    print(methode1([1,1,1], 0.01))
    fin = time.perf_counter()
    duree = fin - debut
    print(f"Durée d'exécution : {duree:.6f} secondes")
