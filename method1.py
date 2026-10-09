import problem as p

def methode1(Xo):
    Xk = Xo 
    norm = 1.0
    dk = [0]*3
    Xkp1 = [0]*3

    while (norm > p.EPS):
        dk = p.gradient(Xk)
        norm = 0.0
        for n in range(3):
            Xkp1[n] = Xk[n] - dk[n]
            norm += dk[n]**2
    
    return Xkp1

print(methode1([1,1,1]))
