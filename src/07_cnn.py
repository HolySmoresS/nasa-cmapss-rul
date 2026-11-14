"""Neural network milestone: a 1D-CNN that reads each engine's last 30 flights.

Questions:
  1. Can a network that sees 30 flights at once match gradient boosting?
  2. Does it still need our hand-made "change from own start" clues?

Everything is decided on the five practice exams (training engines only).
The test set is used once at the end, for the final score.
Run: .venv/bin/python src/07_cnn.py   (about 15 minutes on a laptop)
"""
import matplotlib
matplotlib.use("Agg")  # save plots to files; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from cmapss import (ENGINE_COLORS, PROJECT_ROOT, WINDOW, add_baseline_features, apply_scaler,
                    clip_rul, feature_columns, fit_scaler, load_test_with_rul, load_train,
                    make_windows, plot_predicted_vs_true, predict_cnn, rmse, train_cnn)

N_START, SMOOTH = 20, 10   # same settings as the current best model (src/06)
MAX_EPOCHS = 40            # we watch the curve and pick the best stopping point by cross-validation
SEEDS = [0, 1, 2, 3, 4]    # five different random starts for the final test
FIG = PROJECT_ROOT / "figures"


def new_gb():
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)


def add_feats(df):
    return add_baseline_features(df, N_START, smooth=SMOOTH)


# --- 1. Data ---------------------------------------------------------------
train = add_feats(clip_rul(load_train()))   # labels capped at 125, as before
original = feature_columns(train)            # the 17 original columns
chg = [c for c in train.columns if c.endswith("_chg")]
chg_smooth = [c for c in train.columns if c.endswith(f"_chg_avg{SMOOTH}")]
input_sets = {
    "CNN, 17 original": original,
    "CNN, + change from own start": original + chg + chg_smooth,
}
GB_COLS = original + chg + chg_smooth        # the current best gradient boosting inputs
print(f"{len(original)} original columns; {len(GB_COLS)} with the change clues")

# --- 2. Five practice exams ------------------------------------------------
curves = {name: [] for name in input_sets}   # per fold: validation RMSE after each epoch
train_curves = {name: [] for name in input_sets}
gb_scores = []
folds = list(GroupKFold(n_splits=5).split(train, groups=train["unit"]))
for k, (tr_idx, val_idx) in enumerate(folds, start=1):
    tr, val = train.iloc[tr_idx], train.iloc[val_idx]
    for name, cols in input_sets.items():
        scaler = fit_scaler(tr, cols)    # measured on this fold's training engines only
        Xtr, ytr, _ = make_windows(apply_scaler(tr, cols, scaler), cols)
        Xv, yv, _ = make_windows(apply_scaler(val, cols, scaler), cols)
        _, hist = train_cnn(Xtr, ytr, n_epochs=MAX_EPOCHS, seed=0, X_val=Xv, y_val=yv)
        curves[name].append([h["val"] for h in hist])
        train_curves[name].append([h["train"] for h in hist])
        print(f"fold {k}  {name:30s} val RMSE after {MAX_EPOCHS} epochs: {hist[-1]['val']:.2f}"
              f"  (best {min(h['val'] for h in hist):.2f})", flush=True)
    # Fair comparison: gradient boosting graded on the SAME flights the CNN can answer
    # (flight 30 onward), because the CNN cannot guess before it has 30 flights.
    gb = new_gb().fit(tr[GB_COLS], tr["RUL"])
    same_rows = val[val["cycle"] >= WINDOW]
    gb_scores.append(rmse(same_rows["RUL"], gb.predict(same_rows[GB_COLS])))
    print(f"fold {k}  {'gradient boosting (best so far)':30s} val RMSE: {gb_scores[-1]:.2f}", flush=True)

print("\nPractice-exam average (same flights for every model):")
print(f"  {'gradient boosting (best so far)':32s} {np.mean(gb_scores):.2f}  folds "
      + " ".join(f"{s:.1f}" for s in gb_scores))
best_epochs = {}
for name in input_sets:
    mean_curve = np.mean(curves[name], axis=0)
    best_epochs[name] = int(np.argmin(mean_curve)) + 1
    at_best = [c[best_epochs[name] - 1] for c in curves[name]]
    print(f"  {name:32s} {mean_curve.min():.2f} at epoch {best_epochs[name]}"
          f"  (at epoch {MAX_EPOCHS}: {mean_curve[-1]:.2f})  folds "
          + " ".join(f"{s:.1f}" for s in at_best))

# Learning curves: training error vs held-out error, epoch by epoch.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
epochs = np.arange(1, MAX_EPOCHS + 1)
for ax, name in zip(axes, input_sets):
    ax.plot(epochs, np.mean(train_curves[name], axis=0), color=ENGINE_COLORS[0],
            linewidth=2, label="engines it studied")
    ax.plot(epochs, np.mean(curves[name], axis=0), color=ENGINE_COLORS[1],
            linewidth=2, label="held-out engines")
    ax.axhline(np.mean(gb_scores), color="#52514e", linestyle="--", linewidth=1.2,
               label="gradient boosting, held-out")
    ax.axvline(best_epochs[name], color="#52514e", linestyle=":", linewidth=1)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("epoch (passes through the training clips)")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("RMSE (average of 5 practice exams)")
axes[0].set_ylim(0, 30)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("How the CNN learns, epoch by epoch (dotted line = best stopping point)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "cnn_learning_curves.png", dpi=150)
print("saved figures/cnn_learning_curves.png")

# --- 3. Choose by practice exams, then the test set once --------------------
chosen = min(input_sets, key=lambda n: np.mean(curves[n], axis=0).min())
cols, n_epochs = input_sets[chosen], best_epochs[chosen]
print(f"\nChosen by practice exams: {chosen}, {n_epochs} epochs")

scaler = fit_scaler(train, cols)   # all 100 training engines
Xtr, ytr, _ = make_windows(apply_scaler(train, cols, scaler), cols)
test = add_feats(load_test_with_rul())
Xte, yte, _ = make_windows(apply_scaler(test, cols, scaler), cols, last_only=True)
print(f"test clips: {Xte.shape[0]} (one per engine)")

results = []
for seed in SEEDS:
    model, _ = train_cnn(Xtr, ytr, n_epochs=n_epochs, seed=seed)
    pred = np.clip(predict_cnn(model, Xte), 0, None)   # a negative RUL is impossible
    near = yte <= 30
    results.append({"seed": seed, "pred": pred, "rmse": rmse(yte, pred),
                    "too_high": int((pred > yte).sum()),
                    "near": rmse(yte[near], pred[near]),
                    "near_20_high": int(((pred - yte) > 20)[near].sum())})
    r = results[-1]
    print(f"seed {seed}: test RMSE {r['rmse']:.2f}  too high {r['too_high']}/100  "
          f"near-failure RMSE {r['near']:.2f}  near failure >20 too high: {r['near_20_high']}",
          flush=True)
scores = [r["rmse"] for r in results]
print(f"\nCNN test RMSE over {len(SEEDS)} random starts: average {np.mean(scores):.2f}, "
      f"range {min(scores):.2f} to {max(scores):.2f}")

# Gradient boosting (best so far) on the test set again, for the side-by-side plot.
gb = new_gb().fit(train[GB_COLS], train["RUL"])
test_last = test.groupby("unit").tail(1)
gb_pred = gb.predict(test_last[GB_COLS])
print(f"gradient boosting (best so far) test RMSE: {rmse(test_last['RUL'], gb_pred):.2f}")

typical = sorted(results, key=lambda r: r["rmse"])[len(results) // 2]   # the middle seed, not the luckiest
fig = plot_predicted_vs_true(
    [("Gradient boosting + change clues", test_last["RUL"].to_numpy(), gb_pred),
     (f"1D-CNN ({chosen.split(', ')[1]}), seed {typical['seed']}", yte, typical["pred"])],
    "100 test engines: predicted vs true RUL", limit=200)
fig.savefig(FIG / "cnn_predicted_vs_true.png", dpi=150)
print("saved figures/cnn_predicted_vs_true.png")
