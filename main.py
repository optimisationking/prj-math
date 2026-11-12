import time
import numpy as np
import problem as p
import input as inp
from method1 import methode1
from method2 import methode2
from method3 import methode3
from method4 import methode4

np.seterr(over="ignore", invalid="ignore") 

def run(method):
    start = time.perf_counter()
    Xk, k = method()
    duration = time.perf_counter() - start
    return Xk, k, duration

def status(Xk, grad_norm):
    # Describe how the method ended
    if not grad_norm <= p.EPS:          
        return "pas de convergence"
    if Xk[0] <= 0 or Xk[1] <= 0:        
        return "convergence, r or h <= 0"
    return "convergence"

if __name__ == "__main__":
    X0, alpha = inp.read_input()
    X_star = p.exact_solution()

    print("Paramètres")
    print(f"  V0    = {p.V0}")
    print(f"  X0    = {X0}")
    print(f"  alpha = {alpha}   (méthode 1)")
    print(f"  eps   = {p.EPS}")
    print(f"  KMAX  = {p.KMAX}")
    print(f"  X*    = ({X_star[0]:.8f}, {X_star[1]:.8f}, {X_star[2]:.8f})")
    print()

    # The methods, all called from the same X0
    methods = [
        ("1. Pas fixe",             lambda: methode1(X0, alpha)),
        ("2. Pas optimal",          lambda: methode2(X0)),
        ("3. Newton",               lambda: methode3(X0)),
        ("4. quasi-Newton BFGS",    lambda: methode4(X0))
    ]

    # Comparison table
    header = (f"{'Method':<25} {'Iterations':>10} {'||grad L||':>12} "
              f"{'||X - X*||':>12} {'Time (s)':>10}  {'Status'}")
    print(header)
    print("-" * len(header))

    results = []
    for name, method in methods:
        Xk, k, duration = run(method)
        grad_norm = np.linalg.norm(p.gradient(Xk))
        error = np.linalg.norm(Xk - X_star)
        results.append((name, Xk))
        print(f"{name:<25} {k:>10d} {grad_norm:>12.2e} {error:>12.2e} "
              f"{duration:>10.6f}  {status(Xk, grad_norm)}")

    # Points returned by the methods
    print()
    print("Dernier point X_k = (r, h, lambda)")
    for name, Xk in results:
        print(f"  {name:<25} ({Xk[0]:.8f}, {Xk[1]:.8f}, {Xk[2]:.8f})")