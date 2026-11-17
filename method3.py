import problem as p
import numpy as np
import input as inp
import log as lg
import time

# Méthode 3 : Méthode de Newton (appliquée à grad L(r, h, lambda) = 0)
#
# À chaque itération, on résout le système linéaire 3x3 :
#       Hess L(X_k) d_k = - grad L(X_k)
# puis on met à jour (pas unitaire, alpha_k = 1) :
#       X_{k+1} = X_k + d_k

LOG_FILE = "resultat_methode3.csv"
LOG_COLUMNS = ("alpha", "cond_H")

def methode3(X0):
    X_star = p.exact_solution()
    Xk = np.array(X0, dtype=float)
    gradk = p.gradient(Xk)          # grad L(X_0)
    norme = np.linalg.norm(gradk)   # norme de grad L(X_0)
    cond_H = np.linalg.cond(p.hessienne(Xk))
    k = 0
    rows = [lg.make_row(k, Xk, norme, X_star, (1.0, cond_H))]

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

        cond_H = np.linalg.cond(p.hessienne(Xk))
        rows.append(lg.make_row(k, Xk, norme, X_star, (1.0, cond_H)))

        if not np.isfinite(norme) or np.linalg.norm(Xk) > p.XMAX:    # the iterates diverged: stop
            break

    return Xk, k, rows


if __name__ == "__main__":
    X0, _ = inp.read_input()
    X_star = p.exact_solution()

    print("Méthode 3 : Méthode de Newton")
    print()

    print("Données")
    print("V0   :", p.V0)
    print("X0   :", X0)
    print("eps  :", p.EPS)
    print("KMAX :", p.KMAX)
    print()

    debut = time.perf_counter()
    Xk, k, rows = methode3(X0)
    fin = time.perf_counter()
    duree = fin - debut

    lg.write_log(LOG_FILE, rows, LOG_COLUMNS)

    print("Résultats")
    print("Solution Newton      :", Xk)
    print("Solution exacte      :", X_star)
    print("Itérations           :", k)
    print("||grad L||           :", np.linalg.norm(p.gradient(Xk)))
    print("Erreur               :", np.linalg.norm(Xk - X_star))
    print(f"Durée d'exécution    : {duree:.6f} secondes")
