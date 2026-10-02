import math

V0 = 1
EPS = 1e-3

# X[0] -> r, X[1] -> h, X[2] -> lambda

def lagrengien(X):
    return (math.pi * X[0]**2 + (X[0]**2 + X[1]**2)) - ((math.pi * X[0]**2 * X[1])/3.0) - V0

def gradient(X):
    G = [0] * 3
    G[0] = 4.0 * math.pi**2 * X[0]**3 + 2.0 * math.pi**2 * X[0] * X[1]**2 - (2.0 * X[2] * math.pi * X[0] * X[1]) / 3.0
    G[1] = X[0]**2 * (2.0 * math.pi**2 * X[1] - (X[2] * math.pi) / 3.0)
    G[2] = -1.0 * ((math.pi * X[0]**2 * X[1])/3 - V0)
    return G

X = [1.0, 2.0, 3.0]
print(lagrengien(X))
print(gradient(X))
