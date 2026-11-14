"""Fixing the CNN's memorizing (overfitting).

The first CNN (src/07_cnn.py) learned the training engines almost perfectly
but did worse on new engines, and lost to gradient boosting (17.03 vs 15.18).
Here we try three fixes, one change at a time, on the same five practice exams:
  - dropout: randomly switch off 30% of signals while learning
  - smaller network: average the stencil results instead of remembering positions
  - both together
Then, decided BEFORE looking at the test set: also average the predictions of
the 5 networks trained from different random starts ("ensemble").
Run: .venv/bin/python src/08_cnn_overfitting.py   (about 30 minutes)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold

from cmapss import (ENGINE_COLORS, PROJECT_ROOT, add_baseline_features, apply_scaler, clip_rul,
                    feature_columns, fit_scaler, load_test_with_rul, load_train, make_cnn,
                    make_windows, plot_predicted_vs_true, predict_cnn, rmse, train_cnn)

N_START, SMOOTH = 20, 10
MAX_EPOCHS = 60            # dropout and smaller networks usually need more passes
SEEDS = [0, 1, 2, 3, 4]
FIG = PROJECT_ROOT / "figures"
versions = {               # name -> options for make_cnn
    "first CNN (no fix)": {},
    "+ dropout": {"dropout": 0.3},
    "smaller network": {"pool": True},
    "smaller + dropout": {"pool": True, "dropout": 0.3},
}


def add_feats(df):
    return add_baseline_features(df, N_START, smooth=SMOOTH)


train = add_feats(clip_rul(load_train()))
original = feature_columns(train)
cols = (original + [c for c in train.columns if c.endswith("_chg")]
        + [c for c in train.columns if c.endswith(f"_chg_avg{SMOOTH}")])
for name, net in versions.items():
    print(f"{name:20s} {sum(p.numel() for p in make_cnn(len(cols), **net).parameters()):6d} learned numbers")

# --- Five practice exams ---------------------------------------------------
val_curves = {n: [] for n in versions}
train_curves = {n: [] for n in versions}
for k, (tr_idx, val_idx) in enumerate(GroupKFold(n_splits=5).split(train, groups=train["unit"]), 1):
    tr, val = train.iloc[tr_idx], train.iloc[val_idx]
    scaler = fit_scaler(tr, cols)
    Xtr, ytr, _ = make_windows(apply_scaler(tr, cols, scaler), cols)
    Xv, yv, _ = make_windows(apply_scaler(val, cols, scaler), cols)
    for name, net in versions.items():
        _, hist = train_cnn(Xtr, ytr, n_epochs=MAX_EPOCHS, seed=0, X_val=Xv, y_val=yv, **net)
        val_curves[name].append([h["val"] for h in hist])
        train_curves[name].append([h["train"] for h in hist])
        print(f"fold {k}  {name:20s} best held-out {min(h['val'] for h in hist):.2f}", flush=True)

print("\nPractice-exam average (gradient boosting on the same flights: 13.34)")
best_epoch, best_score = {}, {}
for name in versions:
    v, t = np.mean(val_curves[name], axis=0), np.mean(train_curves[name], axis=0)
    i = int(np.argmin(v))
    best_epoch[name], best_score[name] = i + 1, v[i]
    folds = " ".join(f"{c[i]:.1f}" for c in val_curves[name])
    # The "memorizing gap": error on new engines minus error on studied engines.
    print(f"  {name:20s} held-out {v[i]:.2f} at epoch {i + 1:2d}  studied {t[i]:.2f}  "
          f"gap {v[i] - t[i]:.2f}  | at epoch {MAX_EPOCHS}: held-out {v[-1]:.2f}, gap {v[-1] - t[-1]:.2f}"
          f"  | folds {folds}")

fig, axes = plt.subplots(1, len(versions), figsize=(16, 4.4), sharey=True)
ep = np.arange(1, MAX_EPOCHS + 1)
for ax, name in zip(axes, versions):
    ax.plot(ep, np.mean(train_curves[name], axis=0), color=ENGINE_COLORS[0], lw=2, label="engines it studied")
    ax.plot(ep, np.mean(val_curves[name], axis=0), color=ENGINE_COLORS[1], lw=2, label="held-out engines")
    ax.axhline(13.34, color="#52514e", ls="--", lw=1.2, label="gradient boosting, held-out")
    ax.axvline(best_epoch[name], color="#52514e", ls=":", lw=1)
    ax.set_title(f"{name}\nbest held-out {best_score[name]:.2f}", fontsize=11)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("RMSE (average of 5 practice exams)")
axes[0].set_ylim(0, 30)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Does each fix close the gap between studied and new engines?", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "cnn_overfitting_fixes.png", dpi=150)
print("saved figures/cnn_overfitting_fixes.png")

# --- Choose by practice exams, then the test set once ------------------------
chosen = min(versions, key=best_score.get)
net, n_epochs = versions[chosen], best_epoch[chosen]
print(f"\nChosen by practice exams: {chosen}, {n_epochs} epochs")
scaler = fit_scaler(train, cols)
Xtr, ytr, _ = make_windows(apply_scaler(train, cols, scaler), cols)
test = add_feats(load_test_with_rul())
Xte, yte, _ = make_windows(apply_scaler(test, cols, scaler), cols, last_only=True)
near = yte <= 30

preds = []
for seed in SEEDS:
    model, _ = train_cnn(Xtr, ytr, n_epochs=n_epochs, seed=seed, **net)
    p = np.clip(predict_cnn(model, Xte), 0, None)
    preds.append(p)
    print(f"seed {seed}: test RMSE {rmse(yte, p):.2f}  too high {(p > yte).sum()}/100  "
          f"near-failure RMSE {rmse(yte[near], p[near]):.2f}", flush=True)
scores = [rmse(yte, p) for p in preds]
ens = np.mean(preds, axis=0)   # the 5 networks vote: average their guesses
print(f"\nsingle networks: average {np.mean(scores):.2f}, range {min(scores):.2f} to {max(scores):.2f}")
print(f"average of the 5 networks (ensemble): test RMSE {rmse(yte, ens):.2f}  "
      f"too high {(ens > yte).sum()}/100  near-failure RMSE {rmse(yte[near], ens[near]):.2f}  "
      f"near failure >20 too high: {int(((ens - yte) > 20)[near].sum())}")
print("(first CNN: 17.03 average; gradient boosting: 15.18)")

fig = plot_predicted_vs_true([(f"1D-CNN, {chosen}, average of 5 networks", yte, ens)],
                             "100 test engines: predicted vs true RUL", limit=200)
fig.savefig(FIG / "cnn_fixed_predicted_vs_true.png", dpi=150)
print("saved figures/cnn_fixed_predicted_vs_true.png")
