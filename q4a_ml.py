"""Q4(a): synthetic data, Small MLP regressor, RMSE / R2, 3x3 time predictions."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from common import SEED, OUT, save_json

FEATURES = ["task_size", "crew_experience", "travel_distance", "tool_availability"]


def true_time(size, exp, dist, tool):
    return (2 + 1.2 * size - 0.35 * exp + 0.06 * dist + 3 * (1 - tool)
            + 4 * size / (1 + exp))


def main():
    rng = np.random.default_rng(SEED)
    n = 600
    size = rng.uniform(1, 10, n)
    exp = rng.uniform(1, 10, n)
    dist = rng.uniform(1, 50, n)
    tool = rng.uniform(0, 1, n)
    y = np.clip(true_time(size, exp, dist, tool) + rng.normal(0, 0.3, n), 0.5, None)
    X = np.column_stack([size, exp, dist, tool])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED)
    model = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=3000,
                                       random_state=SEED))
    model.fit(Xtr, ytr)
    pte, ptr = model.predict(Xte), model.predict(Xtr)
    res = {
        "n_rows": n, "n_train": len(ytr), "n_test": len(yte),
        "rmse_test": float(np.sqrt(mean_squared_error(yte, pte))),
        "r2_test": float(r2_score(yte, pte)),
        "rmse_train": float(np.sqrt(mean_squared_error(ytr, ptr))),
        "r2_train": float(r2_score(ytr, ptr)),
    }
    print(f"Train: RMSE={res['rmse_train']:.3f} h  R2={res['r2_train']:.4f}")
    print(f"Test : RMSE={res['rmse_test']:.3f} h  R2={res['r2_test']:.4f}")

    # --- the 3 crews x 3 tasks scenario (fixed, hand-designed feature table) ---
    task_size = [3.0, 6.0, 9.0]
    crew_exp = [9.0, 5.0, 2.0]
    dist = np.array([[10, 30, 45], [35, 8, 25], [20, 40, 6]], float)    # crew i -> task j (km)
    tool = np.array([[0.9, 0.5, 0.8], [0.4, 0.95, 0.6], [0.7, 0.3, 1.0]])
    feats, T = [], np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            feats.append([task_size[j], crew_exp[i], dist[i, j], tool[i, j]])
    T = model.predict(np.array(feats)).reshape(3, 3)
    T_true = np.array([[true_time(task_size[j], crew_exp[i], dist[i, j], tool[i, j])
                        for j in range(3)] for i in range(3)])
    print("Predicted time matrix T[crew][task] (h):\n", np.round(T, 2))
    print("Noise-free true matrix (h):\n", np.round(T_true, 2))
    for j in range(3):
        print(f"Task {j+1}: predictions per crew = {np.round(T[:, j], 2)} -> best crew alone: "
              f"Crew {int(np.argmin(T[:, j]))+1}")
    res.update({"T_true": T_true.tolist(), "task_size": task_size, "crew_exp": crew_exp,
                "distance": dist.tolist(), "tool": tool.tolist()})
    save_json("predictions.json", {"T": T.tolist(), **res})

    # --- plots ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(yte, pte, s=14, alpha=.7)
    lim = [min(yte.min(), pte.min()), max(yte.max(), pte.max())]
    ax[0].plot(lim, lim, "r--"); ax[0].set(xlabel="Actual time (h)", ylabel="Predicted time (h)",
                                          title=f"Test: RMSE={res['rmse_test']:.2f}, R²={res['r2_test']:.3f}")
    ax[1].plot(model[-1].loss_curve_); ax[1].set(xlabel="Iteration", ylabel="Loss", title="MLP training loss",
                                                 yscale="log")
    im = ax[2].imshow(T, cmap="viridis")
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, f"{T[i,j]:.1f}", ha="center", va="center", color="w")
    ax[2].set(xticks=range(3), yticks=range(3), xticklabels=["Task1", "Task2", "Task3"],
              yticklabels=["Crew1", "Crew2", "Crew3"], title="Predicted hours T[crew][task]")
    plt.colorbar(im, ax=ax[2]); plt.tight_layout()
    plt.savefig(f"{OUT}/a_ml.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
