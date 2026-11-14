"""Follow-up: WHY does the best model overestimate some engines near failure?

Run from the project folder:   .venv/bin/python src/06_engine_baselines.py
Outputs: printed results + figures/followup_*.png   (takes a few minutes)
The same steps, with explanations, are in notebooks/05_engine_baselines.ipynb.
"""
import warnings

import matplotlib

warnings.filterwarnings("ignore", message=".*encountered in matmul")
matplotlib.use("Agg")  # draw to an image file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from cmapss import (ENGINE_COLORS, PROJECT_ROOT, RUL_CAP, add_baseline_features, clip_rul,
                    cross_validate_by_engine, feature_columns, load_test_last_rows,
                    load_train, plot_predicted_vs_true, rmse, starting_score)

FIG_DIR = PROJECT_ROOT / "figures"
N_START, SMOOTH = 20, 10   # flights used for the starting level; flights averaged for smoothing

train = load_train()
train_c = clip_rul(train)
features = feature_columns(train)          # the original 17 clues


def new_gb():
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)


def held_out_predictions(df, cols):
    """Predict every training row using a model that never saw that row's engine."""
    out = np.zeros(len(df))
    for tr_idx, val_idx in GroupKFold(n_splits=5).split(df, groups=df["unit"]):
        model = new_gb().fit(df.iloc[tr_idx][cols], df.iloc[tr_idx]["RUL"])
        out[val_idx] = model.predict(df.iloc[val_idx][cols])
    return out


# The three sets of clues we compare.
train_b = add_baseline_features(train_c, N_START, smooth=SMOOTH)
chg = [c for c in train_b.columns if c.endswith("_chg")]
chg_smooth = [c for c in train_b.columns if c.endswith(f"_chg_avg{SMOOTH}")]
clue_sets = {
    "Original 17 clues": features,
    "+ change from own start": features + chg,
    "+ change, also smoothed": features + chg + chg_smooth,
}

# ---------------------------------------------------------------- Step 1
# The cause, measured on TRAINING engines only (the test set is not involved).
# Group engines by how healthy they look at the START of life, then see how far
# off the model is when those same engines are 15 to 30 flights from failure.
start = starting_score(train)
near = ((train_c["RUL"] >= 15) & (train_c["RUL"] <= 30)).to_numpy()
quartile = pd.qcut(start, 4, labels=["lowest", "low", "high", "healthiest-looking"])
miss_by_group = {}
print("Step 1. Average miss near failure (15-30 flights left), training engines graded while held out")
print(f"  {'starting level of the engine':30s}" + "".join(f"{q:>20s}" for q in quartile.cat.categories))
for name in ("Original 17 clues", "+ change from own start"):
    pred = held_out_predictions(train_b, clue_sets[name])
    per_engine = (pd.Series(pred - train_c["RUL"].to_numpy())[near]
                  .groupby(train_c["unit"].to_numpy()[near]).mean())
    miss_by_group[name] = per_engine.groupby(quartile[per_engine.index], observed=True).mean()
    print(f"  {name:30s}" + "".join(f"{v:+20.1f}" for v in miss_by_group[name]))
    print(f"      link between starting level and miss (correlation): "
          f"{np.corrcoef(per_engine, start[per_engine.index])[0, 1]:+.2f}")

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(4)
for i, (name, vals) in enumerate(miss_by_group.items()):
    bars = ax.bar(x + i * 0.4, vals, width=0.37, color=ENGINE_COLORS[i], label=name)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.4 if v >= 0 else -1.3), f"{v:+.1f}",
                ha="center", fontsize=10)
ax.axhline(0, color="#52514e", linewidth=1)
ax.set_ylim(bottom=-6)  # room for the label under the one negative bar
ax.set_xticks(x + 0.2)
ax.set_xticklabels(quartile.cat.categories)
ax.set_xlabel("how healthy the engine's sensors looked in its first 20 flights (25 engines per group)")
ax.set_ylabel("average miss near failure, in flights\n(above 0 = guessed too high)")
ax.set_title("Engines that start out looking healthy get overestimated near failure", fontsize=12)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=9, loc="upper left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "followup_starting_level.png", dpi=120)

# ---------------------------------------------------------------- Step 2
# Does the fix help in general? Five practice exams (cross-validation by engine).
print("\nStep 2. Five practice exams (cross-validation by engine)")
for name, cols in clue_sets.items():
    mean, folds = cross_validate_by_engine(new_gb, train_b, cols)
    print(f"  {name:28s} average RMSE {mean:.2f}   (5 exams: "
          + ", ".join(f"{f:.1f}" for f in folds) + ")")

# ---------------------------------------------------------------- Step 3
# Final exam on the test engines. The new clues are built on each test engine's
# full recorded history first, then we keep its last flight.
test_last = load_test_last_rows(
    add_features=lambda df: add_baseline_features(df, N_START, smooth=SMOOTH))
truth = test_last["RUL"]
near_t = (truth <= 30).to_numpy()
preds = {}
print("\nStep 3. Test set (100 engines)")
print(f"  {'clues':28s} {'RMSE':>6s} {'too high':>9s} {'RMSE near failure':>18s} {'near, >20 too high':>19s}")
for name, cols in clue_sets.items():
    preds[name] = new_gb().fit(train_b[cols], train_b["RUL"]).predict(test_last[cols])
    err = preds[name] - truth
    print(f"  {name:28s} {rmse(truth, preds[name]):6.2f} {(err > 0).sum():9d} "
          f"{rmse(truth[near_t], preds[name][near_t]):18.2f} {(err[near_t] > 20).sum():19d}")

print("\n  The five engines from Week 4 (true RUL -> guess before -> guess after):")
final = "+ change, also smoothed"
for unit in (41, 37, 18, 64, 61):
    i = int(np.where(test_last["unit"] == unit)[0][0])
    print(f"    engine {unit:3d}   true {int(truth[i]):2d}   before {preds['Original 17 clues'][i]:.0f}"
          f"   after {preds[final][i]:.0f}")

fig = plot_predicted_vs_true(
    [(f"Before: original 17 clues", truth, preds["Original 17 clues"]),
     (f"After: + change from own start, smoothed", truth, preds[final])],
    f"Gradient boosting (clipped at {RUL_CAP}) on the 100 test engines", limit=200)
fig.savefig(FIG_DIR / "followup_predicted_vs_true.png", dpi=120)
print("\nSaved: figures/followup_starting_level.png, figures/followup_predicted_vs_true.png")
