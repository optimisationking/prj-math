import problem as p
import numpy as np
import time

# Méthode 3 : Méthode de Newton (appliquée à grad L(r, h, lambda) = 0)
#
# À chaque itération, on résout le système linéaire 3x3 :
#       Hess L(X_k) d_k = - grad L(X_k)
# puis on met à jour (pas unitaire, alpha_k = 1) :
#       X_{k+1} = X_k + d_k


def hessienne(X):
    """Hessienne de L(r, h, lambda) par rapport à (r, h, lambda) (matrice 3x3).

    L = pi^2 r^2 (r^2 + h^2) - lam * (pi/3 r^2 h - V0)
    """
    r, h, lam = X
    pi = p.PI
    d2_rr = pi**2 * (12.0 * r**2 + 2.0 * h**2) - 2.0 * pi * lam * h / 3.0
    d2_rh = 4.0 * pi**2 * r * h - 2.0 * pi * lam * r / 3.0
    d2_rl = -2.0 * pi * r * h / 3.0
    d2_hh = 2.0 * pi**2 * r**2
    d2_hl = -pi * r**2 / 3.0
    return np.array([[d2_rr, d2_rh, d2_rl],
                     [d2_rh, d2_hh, d2_hl],
                     [d2_rl, d2_hl, 0.0]])


def methode3(X0):
    Xk = np.array(X0, dtype=float)
    gradk = p.gradient(Xk)          # grad L(X_0)
    norme = np.linalg.norm(gradk)   # norme de grad L(X_0)
    k = 0

    while norme > p.EPS and k < p.KMAX:
        Hk = hessienne(Xk)
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
