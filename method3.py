import problem as p
import numpy as np
import time

# Méthode 3 : Méthode de Newton (appliquée à grad L(r, h, lambda) = 0)
#
# À chaque itération, on résout le système linéaire 3x3 :
#       Hess L(X_k) d_k = - grad L(X_k)
# puis on met à jour (pas unitaire, alpha_k = 1) :
#       X_{k+1} = X_k + d_k

def methode3(X0):
    Xk = np.array(X0, dtype=float)
    gradk = p.gradient(Xk)          # grad L(X_0)
    norme = np.linalg.norm(gradk)   # norme de grad L(X_0)
    k = 0

    while norme > p.EPS and k < p.KMAX:
        Hk = p.hessienne(Xk)
        try:
            dk = np.linalg.solve(Hk, -gradk)   # direction de Newton
        except np.linalg.LinAlgError:
            print(f"Hessienne singulière à l'itération {k} : arrêt.")
            break
        Xk = Xk + dk                    # pas unitaire (alpha_k = 1)
        gradk = p.gradient(Xk)          # grad L(X_k+1)
        norme = np.linalg.norm(gradk)   # norme de grad L(X_k+1)
        k += 1

    return Xk, k


if __name__ == "__main__":
    debut = time.perf_counter()
    Xk, k = methode3([1, 1, 1])
    fin = time.perf_counter()
    duree = fin - debut

    h_etoile = (6 * p.V0 / p.PI) ** (1 / 3)
    X_etoile = np.array([h_etoile / np.sqrt(2), h_etoile, 6 * p.PI * h_etoile])

    print("Solution Newton :", Xk)
    print("Solution exacte :", X_etoile)
    print("Itérations      :", k)
    print("||grad L||      :", np.linalg.norm(p.gradient(Xk)))
    print("Erreur          :", np.linalg.norm(Xk - X_etoile))
    print(f"Durée d'exécution : {duree:.6f} secondes")
