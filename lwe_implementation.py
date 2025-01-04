import numpy as np
from discrete_gaussians import generate_samples

def F(vector, q, t):
    return np.list([(np.round((q/t) * a)) % q for a in vector])

def F_inverse(vector, q, t):
    return np.list([(np.round((t/q) * a)) % t for a in vector])

def A(n, m, q):    
    return np.random.randint(0, q, size=(n, m))

def E(q, m, l, beta):
    return np.array([generate_samples(beta, q, l) for _ in range(m)])

def private_key_S(q, l, n):
    return np.random.randint(0, q, size=(n, l))

def public_key_P(matrixA, S, E, q):
    return np.mod(np.dot(np.transpose(matrixA), S) + E, q)