"""Shared helpers for the NASA C-MAPSS FD001 project.

Both the Week 1 script (src/02_week1.py) and the notebook (notebooks/01_week1.ipynb)
import these, so every fix happens in ONE place instead of two.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Find data/ relative to this file, so the code works no matter which folder
# you run it from (terminal at the project root, or a notebook in notebooks/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# The raw files have no header row, so we name the 26 columns ourselves.
# op1-op3 = flight settings (altitude, speed, throttle); s1-s21 = sensors.
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_raw(name):
    """Read one space-separated C-MAPSS file, e.g. load_raw('train_FD001.txt')."""
    # sep=r"\s+" means "split on any run of spaces" (the files use uneven spacing).
    return pd.read_csv(DATA_DIR / name, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(df):
    """Add the label column RUL = (engine's last cycle) - (current cycle).

    This only works for TRAINING data, because there every engine runs all the
    way to failure, so its last recorded cycle IS the failure point.
    """
    df = df.copy()  # don't secretly modify the caller's table
    # groupby("unit") = "do this separately for each engine";
    # transform("max") = give every row its own engine's last cycle number.
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = last_cycle - df["cycle"]
    return df


def load_train():
    """Training data with RUL labels attached."""
    return add_rul(load_raw("train_FD001.txt"))


def load_test_last_rows():
    """The LAST row of each test engine, plus its true RUL from RUL_FD001.txt.

    Test engines are cut off at some point before failure. The official
    question is: "standing at the last recorded cycle, how many cycles are left?"
    So we grade ONLY on each engine's final row (100 rows, one per engine).
    """
    test = load_raw("test_FD001.txt")
    # The rows are in order, so the last row per engine is its final snapshot.
    last_rows = test.groupby("unit").tail(1).reset_index(drop=True)
    # RUL_FD001.txt has one number per line: line 1 = engine 1, line 2 = engine 2...
    true_rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["RUL"])
    last_rows["RUL"] = true_rul["RUL"].values
    return last_rows


def constant_columns(df, tol=1e-8):
    """Columns that (almost) never change. They carry no information about wear."""
    candidates = [c for c in COLUMNS if c not in ("unit", "cycle")]
    return [c for c in candidates if df[c].std() < tol]


def rmse(y_true, y_pred):
    """Root Mean Squared Error: the typical size of our mistake, in cycles.

    Steps: take each error, square it (so misses in either direction count and
    big misses count extra), average them, then square-root to get back to cycles.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def sensor_correlations(train):
    """Correlation (-1 to +1) of each non-constant sensor with RUL, strongest first.

    Near 0 = unrelated to wear; near +-1 = moves in lockstep with wear.
    Uses TRAINING data only, so there is no peeking at the test answers.
    """
    const = constant_columns(train)
    usable = [c for c in train.columns if c.startswith("s") and c not in const]
    return train[usable].corrwith(train["RUL"]).sort_values(key=abs, ascending=False)


# Fixed colors, one per engine (colorblind-checked palette). Only 3 engines,
# because more lines than that in small panels gets hard to tell apart.
ENGINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def plot_wear(train, n_sensors=6):
    """Plot the most wear-sensitive sensors for 3 engines (shortest, middle, longest life).

    x-axis = RUL, so every engine's failure point lines up at 0 on the right.
    Returns the matplotlib figure (the caller decides whether to save or show it).
    """
    import matplotlib.pyplot as plt  # imported here so loading data doesn't need it

    corr = sensor_correlations(train)
    sensors = list(corr.index[:n_sensors])
    life = train.groupby("unit")["cycle"].max().sort_values()
    engines = [int(life.index[i]) for i in (0, len(life) // 2, -1)]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, sensor in zip(axes.flat, sensors):
        for color, unit in zip(ENGINE_COLORS, engines):
            e = train[train["unit"] == unit]
            ax.plot(e["RUL"], e[sensor], color=color, linewidth=1.5,
                    label=f"engine {unit} ({life[unit]} cycles)")
        ax.set_title(f"{sensor}  (corr with RUL {corr[sensor]:+.2f})", fontsize=11)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("cycles until failure (RUL)")
    # Flip the x-axis ONCE so time flows left-to-right toward failure (0 on the right).
    # The panels share one x-axis, so flipping inside the loop would flip it 6 times
    # and cancel out (a bug we hit on the first try).
    axes[0, 0].invert_xaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Six most wear-sensitive sensors, three FD001 training engines", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))  # leave room at the top for the legend
    return fig


def feature_columns(df):
    """The columns a model is allowed to use as clues (features).

    We leave out: unit and cycle (ID numbers, not measurements), RUL (the answer!),
    and the constant columns (they never change, so they carry no information).
    """
    skip = ["unit", "cycle", "RUL"] + constant_columns(df)
    return [c for c in COLUMNS if c not in skip]


def split_by_engine(df, val_fraction=0.2, seed=42):
    """Split rows into (train_part, validation_part), keeping each ENGINE whole.

    Why by engine and not by row: flights 100 and 101 of one engine are near-twins.
    If they landed on opposite sides of the split, the model would be "tested" on
    rows it has practically already seen (data leakage) and the score would be fake.
    """
    rng = np.random.default_rng(seed)  # seed = same shuffle every run (repeatable)
    engines = df["unit"].unique()
    rng.shuffle(engines)
    n_val = int(round(len(engines) * val_fraction))
    val_engines = set(engines[:n_val])
    is_val = df["unit"].isin(val_engines)
    return df[~is_val].copy(), df[is_val].copy()


def plot_predicted_vs_true(panels, suptitle):
    """Scatter charts of predicted vs true RUL, one panel per (title, truth, pred).

    Each dot is one prediction. Dots ON the dashed diagonal = perfect guesses;
    above it = guessed too high; below it = guessed too low.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, truth, pred) in zip(axes, panels):
        many = len(truth) > 500  # lots of dots -> make them small and see-through
        ax.scatter(truth, pred, s=8 if many else 28, alpha=0.15 if many else 0.9,
                   color=ENGINE_COLORS[0], linewidths=0)
        ax.plot([0, 360], [0, 360], color="#52514e", linewidth=1.2, linestyle="--")
        ax.set_title(f"{title}\nRMSE {rmse(truth, pred):.1f}", fontsize=11)
        ax.set_xlabel("true RUL (cycles)")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("predicted RUL (cycles)")
    axes[0].annotate("dashed line = perfect prediction", xy=(250, 250), xytext=(150, 330),
                     fontsize=9, color="#52514e",
                     arrowprops={"arrowstyle": "-", "color": "#52514e", "linewidth": 0.8})
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig
