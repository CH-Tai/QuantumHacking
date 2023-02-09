"""
NMR parameters for 3 qubits (generalized to n qubits, shown in the comments)
"""
import warnings
import torch
import math
from pathlib import Path
import matplotlib.pyplot as plt

from core import QOC, train

# ignore complex warnings
warnings.filterwarnings("ignore")

def benchmark(n, N):
    dt = 0.05

    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    Y = torch.tensor([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=torch.complex64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    I = torch.eye(2, dtype=torch.complex64)

    # For defining n-qubit drift & control Hamiltonian
    def hat(gate, index):
        hatgate = torch.tensor([1], dtype=torch.complex64)
        for i in range(n):
            if i != index:
                unitary = I
            if i == index:
                unitary = gate
            hatgate = torch.kron(hatgate, unitary)
        return hatgate

    Xs = torch.stack(list(hat(X, i) for i in range(n)))
    Ys = torch.stack(list(hat(Y, i) for i in range(n)))
    Zs = torch.stack(list(hat(Z, i) for i in range(n)))

    F = torch.zeros(n, n, dtype=torch.complex64)/2
    for i in range(n):
        for j in range(n):
            if j == i + 1:
                F[i][j] = 1

    Hd = torch.einsum("ij, ikl, jlm -> km", F, Zs, Zs)
    Hc = torch.stack([torch.einsum("ijk -> jk", Xs), torch.einsum("ijk -> jk", Ys)])
    maxpower = 1

    # Target unitary
    H = torch.tensor([[1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]], dtype=torch.complex64)
    U0 = torch.tensor([1], dtype = torch.complex64)
    for i in range(n):
        U0 = torch.kron(U0, H)

    model = QOC(Hd, Hc, dt, N, maxpower, torch.randn(Hc.shape[0], N))
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)
    return train(model = model, optim = optim, target = U0, accuracy = 0.01, roughness = 1, weight = 0*1/N, maxiterations = 100000)
