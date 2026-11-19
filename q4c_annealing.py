"""Q4(c): simulated quantum annealing (exact 9-qubit statevector, Trotterised)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

TOTAL_TIME = 400.0
STEPS = 4000
SHOTS = 1000


def anneal(Hp, total_time=TOTAL_TIME, steps=STEPS, record=False):
    """H(s) = (1-s)(-sum X) + s*Hp ; s = t/T linear; returns final psi (+ ground prob trace)."""
    psi = plus_state()
    dt = total_time / steps
    gs = int(np.argmin(Hp))
    trace = []
    for n in range(steps):
        s = (n + 0.5) / steps
        psi = psi * np.exp(-1j * dt * s * Hp)                 # problem part (diagonal)
        psi = apply_mixer(psi, -dt * (1 - s))                 # driver -sum X: exp(+i dt(1-s) sum X)
        if record:
            trace.append((s, abs(psi[gs]) ** 2))
    return psi, trace


def main():
    T = load_T()
    qb = load_json("qubo.json")
    A, E = qb["A"], np.load(f"{OUT}/energies.npy")
    Hp = (E - E.min()) / A                                    # normalised problem Hamiltonian
    psi, trace = anneal(Hp, record=True)
    p = np.abs(psi) ** 2
    p /= p.sum()
    rng = np.random.default_rng(SEED)
    shots = rng.choice(NS, size=SHOTS, p=p)
    counts = np.bincount(shots, minlength=NS)
    top = np.argsort(-counts)[:5]
    X = all_states()
    valid = np.array([is_valid(x) for x in X])
    best = int(top[0])
    bf = qb["best_state"]
    print(f"Anneal time T={TOTAL_TIME}, steps={STEPS}, shots={SHOTS}")
    rows = []
    for s in top:
        rows.append({"bits": bitstring(int(s)), "count": int(counts[s]), "freq": counts[s] / SHOTS,
                     "energy": float(E[s]), "valid": bool(valid[s]),
                     "assignment": [(i + 1, j + 1) for i, j in decode(int(s))]})
        print(rows[-1])
    res = {"total_time": TOTAL_TIME, "steps": STEPS, "shots": SHOTS,
           "best_bits": bitstring(best), "best_energy": float(E[best]), "best_count": int(counts[best]),
           "best_freq": counts[best] / SHOTS, "best_assignment": [(i + 1, j + 1) for i, j in decode(best)],
           "exact_prob_ground": float(p[bf]), "prob_valid": float(p[valid].sum()),
           "matches_bruteforce": best == bf, "bruteforce_bits": qb["best_bits"],
           "bruteforce_energy": qb["best_energy"], "top5": rows}
    print(f"Exact ground-state probability {p[bf]:.4f}; mass on valid permutations {p[valid].sum():.4f}")
    print("Matches brute force:", res["matches_bruteforce"])
    # shorter anneal for contrast
    for tt in (5.0, 50.0, 400.0):
        pp = np.abs(anneal(Hp, tt, STEPS)[0]) ** 2
        res[f"ground_prob_T{int(tt)}"] = float(pp[bf] / pp.sum())
        print(f"  T={tt:>5}: P(ground)={res[f'ground_prob_T{int(tt)}']:.4f}")
    save_json("annealing.json", res)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ids = np.argsort(-counts)[:10]
    cols = ["tab:green" if i == bf else ("tab:blue" if valid[i] else "tab:gray") for i in ids]
    ax[0].bar(range(len(ids)), counts[ids] / SHOTS, color=cols)
    ax[0].set_xticks(range(len(ids))); ax[0].set_xticklabels([bitstring(int(i)) for i in ids], rotation=70, fontsize=7)
    ax[0].set(ylabel="Frequency", title=f"Top outcomes of {SHOTS} shots (green = brute-force optimum)")
    t = np.array(trace)
    ax[1].plot(t[:, 0], t[:, 1]); ax[1].set(xlabel="s = t/T", ylabel="P(ground state)", title="Ground-state probability during anneal")
    plt.tight_layout(); plt.savefig(f"{OUT}/c_annealing.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
