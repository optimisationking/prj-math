import problem as p
import numpy as np

# Méthode 1 : Gradient à pas fixe 

def methode1(X0, alpha):
    Xk = np.array(X0, dtype=float) 
    norm = 1.0
    k = 0

    while norm > p.EPS and k < p.KMAX:
        dk = -p.gradient(Xk)
        Xkp1 = Xk + alpha * dk
        norm = np.linalg.norm(dk)
        Xk = Xkp1
        k += 1
    
    return Xk, k

if __name__ == "__main__":
    print(methode1([1,1,1], 0.01))
