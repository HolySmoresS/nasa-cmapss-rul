"""Week 3: clip RUL at 125, gradient boosting, rolling-window features.

Run from the project folder:   .venv/bin/python src/04_week3.py
Outputs: printed results + figures/week3_*.png
The same steps, with explanations, are in notebooks/03_week3.ipynb.
"""
import warnings

import matplotlib

# numpy 2.0 on Apple chips prints false "overflow in matmul" alarms (verified
# harmless in Week 2 by recomputing the results a second way).
warnings.filterwarnings("ignore", message=".*encountered in matmul")

matplotlib.use("Agg")  # draw to an image file instead of opening a window
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from cmapss import (ENGINE_COLORS, PROJECT_ROOT, RUL_CAP, add_rolling_features, clip_rul,
                    cross_validate_by_engine, feature_columns, load_test_last_rows,
                    load_train, plot_predicted_vs_true, plot_scoreboard, rmse,
                    split_by_engine)

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

train = load_train()                 # labels as they really are (raw)
train_c = clip_rul(train)            # labels capped at 125
features = feature_columns(train)    # the 17 clue columns
test_last = load_test_last_rows()    # last flight of each test engine + true RUL

# ---------------------------------------------------------------- Step 1
# Picture of what clipping does to ONE engine's labels (the longest-lived one).
longest = int(train.groupby("unit")["cycle"].max().idxmax())
e_raw, e_clip = train[train["unit"] == longest], train_c[train_c["unit"] == longest]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(e_raw["cycle"], e_raw["RUL"], color=ENGINE_COLORS[0], linewidth=2,
        linestyle="--", label="raw label: counts down the whole time")
ax.plot(e_clip["cycle"], e_clip["RUL"], color=ENGINE_COLORS[1], linewidth=2,
        label=f"clipped label: flat at {RUL_CAP} ('healthy'), then counts down")
ax.set_xlabel(f"flight number (cycle) of engine {longest}")
ax.set_ylabel("RUL label (flights left)")
ax.set_title("What clipping does to one engine's labels", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "week3_clipping.png", dpi=120)
print(f"Rows whose label changed by clipping: {(train['RUL'] > RUL_CAP).mean():.0%} of training rows")

# Same model as Week 2; the ONLY change is the labels it learns from.
lin_raw = LinearRegression().fit(train[features], train["RUL"])
lin_clip = LinearRegression().fit(train_c[features], train_c["RUL"])
results = {}  # name -> (test RMSE, predictions), filled in as we go
for name, model in [("Linear regression, raw labels", lin_raw),
                    (f"Linear regression, clipped at {RUL_CAP}", lin_clip)]:
    pred = model.predict(test_last[features])
    results[name] = (rmse(test_last["RUL"], pred), pred)
    too_high = (pred > test_last["RUL"]).sum()
    print(f"{name:42s} TEST RMSE {results[name][0]:.2f}   (too high for {too_high} of 100 engines)")

# ---------------------------------------------------------------- Step 2
# Gradient boosting: 100 small trees, each one correcting the mistakes left by
# the trees before it. Small trees (3 questions deep) can't memorize on their own.
def new_gb():
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)

# Overfitting check first, on the same 80/20 engine split as Week 2.
tr, val = split_by_engine(train_c)
gb_check = new_gb().fit(tr[features], tr["RUL"])
lin_check = LinearRegression().fit(tr[features], tr["RUL"])
print("\nOverfitting check (clipped labels, RMSE on studied vs unseen engines):")
for name, m in [("linear regression", lin_check), ("gradient boosting", gb_check)]:
    print(f"  {name:18s} studied {rmse(tr['RUL'], m.predict(tr[features])):.2f}   "
          f"unseen {rmse(val['RUL'], m.predict(val[features])):.2f}")

# Then the real thing: train on all 100 engines, score on the test engines.
gb = new_gb().fit(train_c[features], train_c["RUL"])
pred = gb.predict(test_last[features])
name = f"Gradient boosting, clipped at {RUL_CAP}"
results[name] = (rmse(test_last["RUL"], pred), pred)
print(f"{name:42s} TEST RMSE {results[name][0]:.2f}   "
      f"(too high for {(pred > test_last['RUL']).sum()} of 100 engines)")

# ---------------------------------------------------------------- Step 3
# Rolling-window features. Which window length is best? Let 5 practice exams
# (cross-validation by engine) decide. The test set is NOT used for this choice.
print("\nChoosing the rolling window by 5-fold cross-validation (gradient boosting):")
cv = {}
for window in [None, 5, 10, 20, 30]:
    data = train_c if window is None else add_rolling_features(train_c, window)
    cols = [c for c in data.columns if c not in ("unit", "cycle", "RUL") and c in features
            or "_avg" in c]
    mean, folds = cross_validate_by_engine(new_gb, data, cols)
    cv[window] = mean
    label = "no rolling features" if window is None else f"window = {window:2d} flights"
    print(f"  {label:22s} average RMSE {mean:.2f}   (5 exams: "
          + ", ".join(f"{f:.1f}" for f in folds) + ")")
best = min((w for w in cv if w is not None), key=cv.get)
print(f"  -> best window: {best} flights")

# Final model with the chosen window: train on all 100 engines, score on test.
def add_best(df):
    return add_rolling_features(df, best)

train_r = add_best(train_c)
roll_cols = features + [c for c in train_r.columns if "_avg" in c]
test_last_r = load_test_last_rows(add_features=add_best)
for label, make in [("Linear regression", LinearRegression), ("Gradient boosting", new_gb)]:
    m = make().fit(train_r[roll_cols], train_r["RUL"])
    pred = m.predict(test_last_r[roll_cols])
    name = f"{label}, clipped + rolling avg ({best})"
    results[name] = (rmse(test_last_r["RUL"], pred), pred)
    print(f"{name:42s} TEST RMSE {results[name][0]:.2f}   "
          f"(too high for {(pred > test_last_r['RUL']).sum()} of 100 engines)")

# ---------------------------------------------------------------- Step 4
# Scoreboard of every model so far (the mean baseline is from Week 1).
mean_guess = [train["RUL"].mean()] * len(test_last)
scores = {"Mean guess (Week 1)": rmse(test_last["RUL"], mean_guess)}
scores.update({name: score for name, (score, _) in results.items()})
fig = plot_scoreboard(scores, "FD001 test set: every model so far")
fig.savefig(FIG_DIR / "week3_scoreboard.png", dpi=120)

# Predicted vs true on the 100 test engines, for three models side by side.
show = ["Linear regression, raw labels", f"Linear regression, clipped at {RUL_CAP}",
        f"Gradient boosting, clipped at {RUL_CAP}"]
fig = plot_predicted_vs_true([(n, test_last["RUL"], results[n][1]) for n in show],
                             "Predicted vs true RUL on the 100 test engines", limit=200)
fig.savefig(FIG_DIR / "week3_predicted_vs_true.png", dpi=120)
print("\nSaved: figures/week3_clipping.png, week3_scoreboard.png, week3_predicted_vs_true.png")
