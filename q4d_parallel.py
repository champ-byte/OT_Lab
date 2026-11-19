"""Q4(d): QAOA (p=2) from a grid of (gamma, beta) starting points, run with joblib.Parallel."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_v] = "1"   # keep BLAS single-threaded so workers do not oversubscribe cores
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import minimize
from common import *

P = 2
GRID = 12          # 12 x 12 = 144 starting points
MAXITER = 150


def qaoa_state(params, H):
    psi = plus_state()
    for l in range(P):
        psi = psi * np.exp(-1j * params[2 * l] * H)       # cost layer, gamma_l
        psi = apply_mixer(psi, params[2 * l + 1])         # mixer layer, beta_l
    return psi


def expectation(params, H):
    return float(np.sum(np.abs(qaoa_state(params, H)) ** 2 * H))


def run_start(g, b, H):
    """Local optimisation (COBYLA) from the start point (gamma, beta) repeated on both layers."""
    x0 = np.array([g, b] * P)
    r = minimize(expectation, x0, args=(H,), method="COBYLA", options={"maxiter": MAXITER})
    return float(g), float(b), float(r.fun), r.x.tolist()


def run_grid(H, starts, workers):
    t0 = time.perf_counter()
    out = Parallel(n_jobs=workers)(delayed(run_start)(g, b, H) for g, b in starts)
    return time.perf_counter() - t0, out


def main():
    qb = load_json("qubo.json")
    E = np.load(f"{OUT}/energies.npy")
    A = qb["A"]
    H = (E - E.min()) / A
    gs = np.linspace(0.05, np.pi, GRID)
    bs = np.linspace(0.05, np.pi / 2, GRID)
    starts = [(g, b) for g in gs for b in bs]
    # unoptimised landscape p=1-style energy at start points (for heatmap)
    land = np.array([[expectation(np.array([g, b] * P), H) for b in bs] for g in gs])

    Parallel(n_jobs=2)(delayed(abs)(i) for i in range(4))     # warm-up of the loky pool import
    times, results = {}, {}
    for w in (1, 2, 4):
        runs = []
        for _ in range(5):
            t, out = run_grid(H, starts, w)
            runs.append(t)
        times[w] = runs
        results[w] = out
        print(f"workers={w}: times={np.round(runs, 2)}  (min {min(runs):.2f}s)")
    tmin = {w: min(v) for w, v in times.items()}
    speed = {w: tmin[1] / tmin[w] for w in tmin}
    eff = {w: speed[w] / w for w in tmin}
    for w in tmin:
        print(f"w={w}: time={tmin[w]:.2f}s speedup={speed[w]:.2f} efficiency={eff[w]:.2f}")

    same = all(np.allclose([r[2] for r in results[1]], [r[2] for r in results[w]]) for w in (2, 4))
    best = min(results[1], key=lambda r: r[2])
    psi = qaoa_state(np.array(best[3]), H)
    p = np.abs(psi) ** 2
    p /= p.sum()
    rng = np.random.default_rng(SEED)
    counts = np.bincount(rng.choice(NS, 1000, p=p), minlength=NS)
    top = int(np.argmax(counts))
    res = {"p_layers": P, "grid": GRID, "n_starts": len(starts), "maxiter": MAXITER,
           "times": {str(w): v for w, v in times.items()},
           "time_median": {str(w): float(np.median(v)) for w, v in times.items()},
           "time_min": {str(w): v for w, v in tmin.items()},
           "speedup": {str(w): v for w, v in speed.items()},
           "efficiency": {str(w): v for w, v in eff.items()},
           "identical_results": bool(same), "best_start": [best[0], best[1]],
           "best_expected_energy_norm": best[2], "best_expected_energy": best[2] * A + E.min(),
           "best_params": best[3], "top_bits": bitstring(top), "top_freq": counts[top] / 1000,
           "top_energy": float(E[top]), "top_valid": is_valid(all_states()[top]),
           "top_matches_bruteforce": top == qb["best_state"],
           "prob_ground": float(p[qb["best_state"]]),
           "top_assignment": [(i + 1, j + 1) for i, j in decode(top)],
           "n_cores": os.cpu_count()}
    print({k: v for k, v in res.items() if k not in ("times",)})
    save_json("parallel.json", res)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    im = ax[0].imshow(land.T, origin="lower", aspect="auto", cmap="viridis",
                      extent=[gs[0], gs[-1], bs[0], bs[-1]])
    ax[0].set(xlabel="γ", ylabel="β", title="⟨H⟩ at the (γ,β) grid starting points"); plt.colorbar(im, ax=ax[0])
    ws = list(tmin)
    ax[1].bar([str(w) for w in ws], [tmin[w] for w in ws], color="tab:blue")
    ax[1].set(xlabel="workers", ylabel="time (s)", title="Wall-clock time (best of 5)")
    ax[2].plot(ws, [speed[w] for w in ws], "o-", label="speed-up")
    ax[2].plot(ws, ws, "k--", label="ideal")
    ax[2].plot(ws, [eff[w] for w in ws], "s-", label="efficiency")
    ax[2].set(xlabel="workers", title="Speed-up and efficiency"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/d_parallel.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
