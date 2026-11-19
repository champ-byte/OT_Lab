"""Q4(b): QUBO for the 3x3 assignment, penalty, brute-force verification."""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from common import *


def main():
    T = load_T()
    A = penalty_A(T)
    Q, off = build_qubo(T, A)
    np.save(f"{OUT}/qubo_Q.npy", Q)
    print(f"Penalty A = {A:.2f}, offset = 6A = {off:.2f}\nQ (upper-triangular) =\n", np.round(Q, 2))

    X = all_states()
    E_q = qubo_energies(Q, off)
    E_d = np.array([direct_energy(x, T, A) for x in X])
    print("max |x^T Q x + offset - E(x)| over 512 states:", np.abs(E_q - E_d).max())
    assert np.allclose(E_q, E_d)

    order = np.argsort(E_q)
    valid = np.array([is_valid(x) for x in X])
    best = int(order[0])
    print("\nLowest 8 states:")
    rows = []
    for s in order[:8]:
        rows.append({"bits": bitstring(int(s)), "energy": float(E_q[s]), "valid": bool(valid[s]),
                     "assignment": [(i + 1, j + 1) for i, j in decode(int(s))]})
        print(rows[-1])

    # permutation cross-check
    perms = []
    for p in itertools.permutations(range(3)):
        perms.append((p, sum(T[i, p[i]] for i in range(3))))
    perms.sort(key=lambda t: t[1])
    r, c = linear_sum_assignment(T)
    print("\nAll 6 permutations (crew i -> task p[i]) :")
    for p, t in perms:
        print([f"C{i+1}->T{p[i]+1}" for i in range(3)], f"total {t:.3f} h")
    print("Hungarian:", list(zip(r + 1, c + 1)), T[r, c].sum())
    ok = (is_valid(X[best]) and abs(E_q[best] - perms[0][1]) < 1e-9
          and abs(T[r, c].sum() - perms[0][1]) < 1e-9)
    print("Brute-force ground state is valid optimum permutation:", ok)
    assert ok
    min_invalid = float(E_q[~valid].min())
    print(f"Lowest invalid energy {min_invalid:.3f} > optimum {E_q[best]:.3f}")

    save_json("qubo.json", {"A": A, "offset": off, "Q": Q.tolist(), "best_state": best,
                            "best_bits": bitstring(best), "best_energy": float(E_q[best]),
                            "best_assignment": [(i + 1, j + 1) for i, j in decode(best)],
                            "lowest": rows, "min_invalid_energy": min_invalid,
                            "perms": [[[int(v) + 1 for v in p], float(t)] for p, t in perms],
                            "n_valid": int(valid.sum())})
    np.save(f"{OUT}/energies.npy", E_q)

    srt = np.sort(E_q)
    v_sorted = valid[order]
    plt.figure(figsize=(9, 4))
    plt.scatter(np.arange(NS)[~v_sorted], srt[~v_sorted], s=8, c="tab:gray", label="invalid (rule broken)")
    plt.scatter(np.arange(NS)[v_sorted], srt[v_sorted], s=30, c="tab:red", label="valid permutations (6)")
    plt.xlabel("State rank"); plt.ylabel("QUBO energy"); plt.title("Energy spectrum of all 512 bitstrings")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/b_energy_landscape.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
