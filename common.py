"""Shared helpers for Q.4 (ML + Quantum + Parallel). Seed 081 -> 81."""
import json
import os
import numpy as np

SEED = 81
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
NC = 3            # crews = tasks = 3
NQ = NC * NC      # 9 qubits / binary variables
NS = 2 ** NQ      # 512 basis states


def idx(i, j):
    """Variable index of x_ij (crew i does task j)."""
    return NC * i + j


def save_json(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)


def load_json(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def load_T():
    return np.array(load_json("predictions.json")["T"])


def penalty_A(T):
    return 2.0 * float(T.max())


def build_qubo(T, A):
    """Upper-triangular Q and constant offset so that E(x)=x^T Q x + offset.

    E(x) = sum T_ij x_ij + A sum_i (sum_j x_ij - 1)^2 + A sum_j (sum_i x_ij - 1)^2
    Using x^2 = x:  (sum x - 1)^2 = -sum x + 2 sum_{pairs} x x' + 1
    => diag Q_kk = T_ij - 2A, pair Q_kl = 2A (same crew or same task), offset = 6A.
    """
    Q = np.zeros((NQ, NQ))
    for i in range(NC):
        for j in range(NC):
            Q[idx(i, j), idx(i, j)] = T[i, j] - 2 * A
    for k in range(NQ):
        for l in range(k + 1, NQ):
            (ik, jk), (il, jl) = divmod(k, NC), divmod(l, NC)
            if ik == il or jk == jl:
                Q[k, l] = 2 * A
    return Q, 6 * A


def direct_energy(x, T, A):
    X = np.asarray(x).reshape(NC, NC)
    return (T * X).sum() + A * ((X.sum(1) - 1) ** 2).sum() + A * ((X.sum(0) - 1) ** 2).sum()


def all_states():
    """(512, 9) bit matrix; bit k of state index s is X[s, k]."""
    s = np.arange(NS)
    return ((s[:, None] >> np.arange(NQ)) & 1).astype(int)


def qubo_energies(Q, offset):
    X = all_states()
    return np.einsum("si,ij,sj->s", X, Q, X) + offset


def is_valid(x):
    X = np.asarray(x).reshape(NC, NC)
    return bool((X.sum(0) == 1).all() and (X.sum(1) == 1).all())


def bitstring(s):
    """Bitstring x_0..x_8 (variable order) of state index s."""
    return "".join(str((s >> k) & 1) for k in range(NQ))


def decode(s):
    X = np.array([(s >> k) & 1 for k in range(NQ)]).reshape(NC, NC)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(X))]


# ---- statevector helpers (qubit k = bit k of the state index) ----
_FLIP = [np.arange(NS) ^ (1 << k) for k in range(NQ)]


def apply_mixer(psi, beta):
    """exp(-i beta sum_k X_k) on all qubits."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    for k in range(NQ):
        psi = c * psi + s * psi[_FLIP[k]]
    return psi


def plus_state():
    return np.full(NS, 1 / np.sqrt(NS), dtype=complex)
