"""Week 4: the final results table, and a look at WHERE the best model is still wrong.

Run from the project folder:   .venv/bin/python src/05_week4.py
Outputs: printed results + figures/week4_error_by_stage.png
The same steps, with explanations, are in notebooks/04_week4.ipynb.
"""
import warnings

import matplotlib

# numpy 2.0 on Apple chips prints false "overflow in matmul" alarms (verified
# harmless in Week 2 by recomputing the results a second way).
warnings.filterwarnings("ignore", message=".*encountered in matmul")

matplotlib.use("Agg")  # draw to an image file instead of opening a window
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from cmapss import (PROJECT_ROOT, RUL_CAP, clip_rul, feature_columns, load_test_last_rows,
                    load_train, plot_error_by_stage, rmse, stage_of_life)

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

train = load_train()                 # labels as they really are (raw)
train_c = clip_rul(train)            # labels capped at 125
features = feature_columns(train)    # the 17 clue columns
test_last = load_test_last_rows()    # last flight of each test engine + true RUL
truth = test_last["RUL"]

# ---------------------------------------------------------------- Step 1
# Rebuild the four main models from scratch, so every number in the README's
# results table can be reproduced by running this one file.
def new_gb():
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)

preds = {
    "Mean baseline": np.full(len(test_last), train["RUL"].mean()),
    "Linear regression, raw labels":
        LinearRegression().fit(train[features], train["RUL"]).predict(test_last[features]),
    f"Linear regression, clipped at {RUL_CAP}":
        LinearRegression().fit(train_c[features], train_c["RUL"]).predict(test_last[features]),
    f"Gradient boosting, clipped at {RUL_CAP}":
        new_gb().fit(train_c[features], train_c["RUL"]).predict(test_last[features]),
}

print("Final results on the 100 FD001 test engines")
print(f"  {'model':40s} {'RMSE':>6s} {'too high':>9s} {'worst miss':>11s}")
for name, pred in preds.items():
    err = pred - truth
    print(f"  {name:40s} {rmse(truth, pred):6.2f} {(err > 0).sum():9d} {err.abs().max():11.0f}")

# ---------------------------------------------------------------- Step 2
# One average hides where the mistakes are. Sort the test engines by how much
# life they TRULY have left, and score each group on its own.
best_name = f"Gradient boosting, clipped at {RUL_CAP}"
best = preds[best_name]
stage = stage_of_life(truth)

print(f"\nWhere is the best model wrong? ({best_name})")
print(f"  {'group (true flights left)':30s} {'engines':>8s} {'RMSE':>6s} {'average error':>14s}")
for label in stage.cat.categories:
    m = (stage == label).to_numpy()
    print(f"  {label:30s} {m.sum():8d} {rmse(truth[m], best[m]):6.2f} "
          f"{(best[m] - truth[m]).mean():+14.1f}")

# The "close to failure" group matters most for safety, so look inside it.
# Median = the middle value when the misses are lined up smallest to largest.
# Unlike RMSE, it is not dragged up by a few huge misses.
near = (truth <= 30).to_numpy()
near_err = best[near] - truth[near]
print(f"\nInside the close-to-failure group ({near.sum()} engines):")
print(f"  middle (median) miss: {np.median(np.abs(near_err)):.1f} flights")
print(f"  within 10 flights of the truth: {(np.abs(near_err) <= 10).sum()} engines")
print(f"  too HIGH by more than 20 flights: {(near_err > 20).sum()} engines")
worst = test_last[near].assign(predicted=best[near], miss=near_err).nlargest(5, "miss")
print("  the five worst (engine, flights flown so far, true RUL, predicted):")
for _, r in worst.iterrows():
    print(f"    engine {int(r['unit']):3d}   flown {int(r['cycle']):3d}   "
          f"true {int(r['RUL']):2d}   predicted {r['predicted']:.0f}")

# ---------------------------------------------------------------- Step 3
# How much of the error is BUILT IN by clipping? A model trained on labels capped
# at 125 can never say more than about 125. So imagine a perfect clipped model:
# exactly right below 125, and saying 125 for everything above. Its score is the
# best any clipped model could possibly get on this test set.
perfect_clipped = np.minimum(truth, RUL_CAP)
print(f"\nHighest prediction the best model ever makes: {best.max():.1f}")
print(f"Score of a PERFECT clipped model (the floor): {rmse(truth, perfect_clipped):.2f}")
healthy = (truth > RUL_CAP).to_numpy()
print(f"Best model scored only on the {(~healthy).sum()} engines at or below {RUL_CAP}: "
      f"{rmse(truth[~healthy], best[~healthy]):.2f}")

fig = plot_error_by_stage(truth, {f"Linear regression, clipped at {RUL_CAP}":
                                  preds[f"Linear regression, clipped at {RUL_CAP}"],
                                  best_name: best})
out = FIG_DIR / "week4_error_by_stage.png"
fig.savefig(out, dpi=120)
print(f"\nSaved plot: {out.relative_to(PROJECT_ROOT)}")
