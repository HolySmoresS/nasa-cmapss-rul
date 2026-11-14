"""Week 1: RUL labels, sensor plots, and the mean-guess baseline.

Run from the project folder:   .venv/bin/python src/02_week1.py
Outputs: printed results + figures/week1_sensors.png
The same steps, with explanations, are in notebooks/01_week1.ipynb.
"""
import matplotlib

matplotlib.use("Agg")  # draw to an image file instead of opening a window

from cmapss import (PROJECT_ROOT, constant_columns, load_test_last_rows, load_train,
                    plot_wear, rmse, sensor_correlations)

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Step 2
train = load_train()  # raw training data + RUL label column
print(f"Training data: {train.shape[0]} rows, {train['unit'].nunique()} engines")

# ---------------------------------------------------------------- Step 3a
# Sensors that never change can't tell a new engine from a worn one: drop them.
const = constant_columns(train)
print("\nConstant columns (no information):", const)
features = [c for c in train.columns if c not in ["unit", "cycle", "RUL"] + const]
print(f"Usable feature columns: {len(features)} -> {features}")

# ---------------------------------------------------------------- Step 3b
corr = sensor_correlations(train)
print("\nCorrelation of each sensor with RUL (strongest first):")
print(corr.round(3).to_string())

# ---------------------------------------------------------------- Step 3c
fig = plot_wear(train)
out = FIG_DIR / "week1_sensors.png"
fig.savefig(out, dpi=120)
print(f"\nSaved plot: {out.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------- Step 4
# Mean-guess baseline: ignore the sensors, predict ONE number for every engine.
# The number comes ONLY from training data. Using anything from the test set
# to choose it would be "data leakage" (peeking at the answer key).
mean_rul = train["RUL"].mean()

test_last = load_test_last_rows()  # 100 rows: last snapshot of each test engine
preds = [mean_rul] * len(test_last)
score = rmse(test_last["RUL"], preds)

print(f"\nMean RUL over all training rows: {mean_rul:.1f} cycles")
print(f"Test engines: {len(test_last)};  true RUL ranges {test_last['RUL'].min()}-"
      f"{test_last['RUL'].max()}, average {test_last['RUL'].mean():.1f}")
print(f"Mean-guess baseline test RMSE: {score:.2f} cycles")
