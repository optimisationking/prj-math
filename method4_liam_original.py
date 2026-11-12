import problem as p
import numpy as np
import time

# Méthode 4 : quasi-Newton BFGS (SQP-BFGS)
# Newton appliqué à grad L = 0, où la Hessienne de L (en r, h) est remplacée
# par une matrice B_k 2x2 mise à jour avec la formule BFGS.

def f(x):
    # f(r, h) = pi^2 r^2 (r^2 + h^2)   (c'est S^2)
    r, h = x
    return p.PI**2 * r**2 * (r**2 + h**2)

def c(x):
    # c(r, h) = pi/3 r^2 h - V0   (on veut c = 0)
    r, h = x
    return p.PI * r**2 * h / 3.0 - p.V0

def grad_x_L(x, lam):
    # dérivées de L par rapport à (r, h) seulement (on enlève la 3e composante de p.gradient)
    return p.gradient([x[0], x[1], lam])[:2]

def grad_f(x, lam):
    # grad f = grad_x L + lambda * grad c    (car L = f - lambda * c)
    return grad_x_L(x, lam) + lam * p.constraint_gradient([x[0], x[1], lam])

def direction(x, lam, B):
    # Étape A : résoudre le système de Newton où la Hessienne est remplacée par B
    #   [  B   -a ] [ d    ]   [ -grad_x L ]
    #   [ -a^T  0 ] [ d_lam] = [     c     ]        avec a = grad c
    a = p.constraint_gradient([x[0], x[1], lam])
    K = np.block([[B, -a[:, None]],
                  [-a[None, :], np.zeros((1, 1))]])
    rhs = np.append(-grad_x_L(x, lam), c(x))
    sol = np.linalg.solve(K, rhs)
    return sol[:2], sol[2]          # d = (d_r, d_h), d_lambda

def recherche_lineaire(x, lam, d, lam_new, rho):
    # Étape B : backtracking d'Armijo sur la fonction mérite Phi = f + rho*|c|
    rho = max(rho, abs(lam_new) + 1.0)
    Phi = lambda z: f(z) + rho * abs(c(z)) if np.all(z > 0) else np.inf   # r, h > 0
    pente = grad_f(x, lam) @ d - rho * abs(c(x))
    alpha = 1.0
    while Phi(x + alpha * d) > Phi(x) + 0.1 * alpha * pente and alpha > 1e-12:
        alpha /= 2.0
    return alpha, rho

def update_B(B, s, y):
    # Étape C : mise à jour BFGS (seulement si y^T s > 0, pour que B reste définie positive)
    if y @ s > 1e-12:
        Bs = B @ s
        B = B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / (y @ s)
    return B

def methode4(X0):
    Xk = np.array(X0, dtype=float)
    x, lam = Xk[:2], Xk[2]
    B = np.eye(2)                    # B_0 = I
    rho = 1.0
    gradk = p.gradient(Xk)           # grad L(X_0)
    norme = np.linalg.norm(gradk)
    k = 0

    while norme > p.EPS and k < p.KMAX:
        d, d_lam = direction(x, lam, B)
        lam_new = lam + d_lam
        alpha, rho = recherche_lineaire(x, lam, d, lam_new, rho)

        x_new = x + alpha * d
        s = x_new - x
        y = grad_x_L(x_new, lam_new) - grad_x_L(x, lam_new)   # même lambda (la nouvelle valeur)
        B = update_B(B, s, y)

        x, lam = x_new, lam_new
        Xk = np.append(x, lam)
        gradk = p.gradient(Xk)
        norme = np.linalg.norm(gradk)
        k += 1

        if not np.isfinite(norme):
            break

    return Xk, k

if __name__ == "__main__":
    debut = time.perf_counter()
    Xk, k = methode4([1, 1, 1])
    fin = time.perf_counter()
    duree = fin - debut

    h_etoile = (6 * p.V0 / p.PI) ** (1 / 3)
    X_etoile = np.array([h_etoile / np.sqrt(2), h_etoile, 6 * p.PI * h_etoile])

    print("Solution BFGS   :", Xk)
    print("Solution exacte :", X_etoile)
    print("Itérations      :", k)
    print("||grad L||      :", np.linalg.norm(p.gradient(Xk)))
    print("Erreur          :", np.linalg.norm(Xk - X_etoile))
    print(f"Durée d'exécution : {duree:.6f} secondes")
