"""Week 2: linear regression, with a train/validation split by ENGINE.

Run from the project folder:   .venv/bin/python src/03_week2.py
Outputs: printed results + figures/week2_predicted_vs_true.png
The same steps, with explanations, are in notebooks/02_week2.ipynb.
"""
import warnings

import matplotlib

# numpy 2.0 on Apple chips prints false "divide by zero / overflow in matmul"
# alarms during matrix multiplication. We checked the results by recomputing
# them a second way (identical), so we silence just that one message.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

matplotlib.use("Agg")  # draw to an image file instead of opening a window
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from cmapss import (PROJECT_ROOT, feature_columns, load_test_last_rows, load_train,
                    plot_predicted_vs_true, rmse, split_by_engine)

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Step 1
train = load_train()
features = feature_columns(train)          # the 17 clue columns
tr, val = split_by_engine(train)           # 80 engines to learn from, 20 to check on
print(f"Features: {len(features)}.  Training: {tr['unit'].nunique()} engines, "
      f"{len(tr)} rows.  Validation: {val['unit'].nunique()} engines, {len(val)} rows.")

# ---------------------------------------------------------------- Step 2
# Linear regression: RUL ~ w1*sensor1 + w2*sensor2 + ... + a starting number.
# .fit() = "training": find the weights that make guesses closest to the answers.
linreg = LinearRegression().fit(tr[features], tr["RUL"])

lin_train = rmse(tr["RUL"], linreg.predict(tr[features]))
lin_val = rmse(val["RUL"], linreg.predict(val[features]))
# For comparison: the Week 1 mean guess, scored on the same validation rows.
mean_val = rmse(val["RUL"], [tr["RUL"].mean()] * len(val))

print("\nLinear regression")
print(f"  RMSE on training engines (studied):  {lin_train:.2f}")
print(f"  RMSE on validation engines (unseen): {lin_val:.2f}")
print(f"  (mean guess on the same validation rows: {mean_val:.2f})")

# The weights show what the model learned: sign = direction, size = strength.
# Sensors have very different scales, so we multiply each weight by that sensor's
# typical spread (std) to compare them fairly: "cycles of RUL per typical wiggle".
effect = (linreg.coef_ * tr[features].std()).sort_values(key=abs, ascending=False)
print("\nStrongest clues (RUL change per one typical wiggle of the sensor):")
print(effect.head(5).round(1).to_string())

# ---------------------------------------------------------------- Step 3a
# Luck check: with only 20 validation engines, WHICH 20 you draw matters.
print("\nSame model, 5 different random splits (seed = which shuffle):")
for seed in range(5):
    a, b = split_by_engine(train, seed=seed)
    m = LinearRegression().fit(a[features], a["RUL"])
    print(f"  seed {seed}: train {rmse(a['RUL'], m.predict(a[features])):.1f}   "
          f"validation {rmse(b['RUL'], m.predict(b[features])):.1f}   "
          f"(longest validation engine: {b.groupby('unit')['cycle'].max().max()} cycles)")

# ---------------------------------------------------------------- Step 3b
# Overfitting demo: a decision tree with no size limit can grow one branch per
# training row, i.e. memorize every answer.
tree = DecisionTreeRegressor(random_state=0).fit(tr[features], tr["RUL"])
print("\nUnlimited decision tree (a memorizer)")
print(f"  RMSE on training engines (studied):  {rmse(tr['RUL'], tree.predict(tr[features])):.2f}")
print(f"  RMSE on validation engines (unseen): {rmse(val['RUL'], tree.predict(val[features])):.2f}")

# ---------------------------------------------------------------- Step 3c
# Leakage demo: score the memorizer under a by-ROW split and a by-ENGINE split,
# with three different sets of clues. Leakage shows up as a by-row score that
# looks far better than the honest by-engine score.
row_tr, row_val = train_test_split(train, test_size=0.2, random_state=42)  # rows shuffled
clue_sets = {
    "17 sensor/setting columns only": features,
    "+ flight number (cycle)": features + ["cycle"],
    "+ engine number AND flight number": features + ["unit", "cycle"],
}
print("\nMemorizer's validation RMSE under each kind of split:")
print(f"  {'clues given to the model':36s} {'by ROW':>8s} {'by ENGINE':>10s}")
for name, cols in clue_sets.items():
    t_row = DecisionTreeRegressor(random_state=0).fit(row_tr[cols], row_tr["RUL"])
    t_eng = DecisionTreeRegressor(random_state=0).fit(tr[cols], tr["RUL"])
    by_row = rmse(row_val["RUL"], t_row.predict(row_val[cols]))
    by_eng = rmse(val["RUL"], t_eng.predict(val[cols]))
    print(f"  {name:36s} {by_row:8.2f} {by_eng:10.2f}")

# ---------------------------------------------------------------- Step 4
# Final exam. Retrain on ALL 100 training engines (no more choices to make, so
# the validation group can rejoin), then score ONCE on the test engines' last rows.
final = LinearRegression().fit(train[features], train["RUL"])
test_last = load_test_last_rows()
test_pred = final.predict(test_last[features])
test_score = rmse(test_last["RUL"], test_pred)
print(f"\nLinear regression, raw RUL labels -> TEST RMSE: {test_score:.2f}  (mean baseline: 52.62)")
err = test_pred - test_last["RUL"]
print(f"  predicted too HIGH for {(err > 0).sum()} engines, too LOW for {(err < 0).sum()};"
      f"  worst miss: {err.abs().max():.0f} cycles")

# Chart: one dot per prediction. Dots ON the diagonal line = perfect predictions.
fig = plot_predicted_vs_true(
    [("Validation: every flight of 20 unseen engines", val["RUL"], linreg.predict(val[features])),
     ("Test: last recorded flight of 100 engines", test_last["RUL"], test_pred)],
    "Linear regression: predicted vs true RUL")
out = FIG_DIR / "week2_predicted_vs_true.png"
fig.savefig(out, dpi=120)
print(f"Saved plot: {out.relative_to(PROJECT_ROOT)}")
