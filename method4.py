import problem as p
import numpy as np
import time

# Méthode 4 : quasi-Newton BFGS (SQP-BFGS)
# Newton appliqué à grad L = 0, où la Hessienne de L (en r, h) est remplacée
# par une matrice B_k 2x2 mise à jour avec la formule BFGS.

C_ARMIJO = 0.1          # constant of the Armijo condition
ALPHA_MIN = 1e-12       # smallest step accepted by the line search
YS_MIN = 1e-12          # below this value, BFGS update is skipped

def bfgs_direction(Xk, gradk, Bk):
    # Solve quasi-Newton system Hk dk = -grad L(Xk)
    grad_c = p.constraint_gradient(Xk)
    Hk = np.zeros((3, 3))
    Hk[:2, :2] = Bk 
    Hk[:2, 2] = -grad_c 
    Hk[2, :2] = -grad_c 
    return np.linalg.solve(Hk, -gradk)

def merit(X, rho):
    # Merit function phi = f + rho * |c|
    r, h, _ = X 
    if r <= 0 or h <= 0:
        return np.inf 
    f = p.lagrangien([r, h, 0.0])
    c = -p.gradient(X)[2]
    return f + rho * abs(c)

def armijo_step(Xk, gradk, dk, rho):
    # Armijo backtracking on the merit function
    # Returns the step and the new rho
    lam = Xk[2]
    c = -gradk[2]
    rho = max(rho, abs(lam + dk[2]) + 1.0)

    grad_f = gradk[:2] + lam * p.constraint_gradient(Xk)
    slope = np.dot(grad_f, dk[:2]) - rho * abs(c)
    alpha = 1.0
    phik = merit(Xk, rho)
    while merit(Xk + alpha * dk, rho) > phik + C_ARMIJO * alpha * slope and alpha > ALPHA_MIN:
        alpha /= 2.0

    return alpha, rho

def bfgs_update(Bk, Xk, Xnew):
    # Update Bk with the BFGS formula
    s = Xnew[:2] - Xk[:2]
    X_mixed = np.array([Xk[0], Xk[1], Xnew[2]])
    y = p.gradient(Xnew)[:2] - p.gradient(X_mixed)[:2]

    if np.dot(y, s) <= YS_MIN:
        return Bk 

    Bs = np.dot(Bk, s)
    return Bk - np.outer(Bs, Bs) / np.dot(s, Bs) + np.outer(y, y) / np.dot(y, s)

def update_X(Xk, ak, dk):
    # (r, h) moves with step ak, lambda takes the full step
    Xnew = Xk + ak * dk
    Xnew[2] = Xk[2] + dk[2]
    return Xnew

def methode4(X0):
    Xk = np.array(X0, dtype=float)
    gradk = p.gradient(Xk)
    norm = np.linalg.norm(gradk)
    Bk = np.eye(2) 
    rho = 1.0 
    k = 0

    while norm > p.EPS and k < p.KMAX:
        dk = bfgs_direction(Xk, gradk, Bk)
        ak, rho = armijo_step(Xk, gradk, dk, rho)
        Xnew = update_X(Xk, ak, dk)
        Bk = bfgs_update(Bk, Xk, Xnew)
        Xk = Xnew 
        gradk = p.gradient(Xk)
        norm = np.linalg.norm(gradk)
        k += 1

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
