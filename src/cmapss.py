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


def load_test_last_rows(add_features=None):
    """The LAST row of each test engine, plus its true RUL from RUL_FD001.txt.

    Test engines are cut off at some point before failure. The official
    question is: "standing at the last recorded cycle, how many cycles are left?"
    So we grade ONLY on each engine's final row (100 rows, one per engine).
    """
    test = load_raw("test_FD001.txt")
    if add_features is not None:
        # Features like rolling averages need each engine's EARLIER flights, so we
        # build them on the full test history first and only then keep the last row.
        test = add_features(test)
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


def plot_predicted_vs_true(panels, suptitle, limit=360):
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
        ax.plot([0, limit], [0, limit], color="#52514e", linewidth=1.2, linestyle="--")
        ax.set_title(f"{title}\nRMSE {rmse(truth, pred):.1f}", fontsize=11)
        ax.set_xlabel("true RUL (cycles)")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("predicted RUL (cycles)")
    # Plain caption in the empty top-left corner (no arrow, so it never crosses dots).
    axes[0].text(0.03, 0.96, "dashed line = perfect prediction", transform=axes[0].transAxes,
                 fontsize=9, color="#52514e", va="top")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


RUL_CAP = 125  # "more than this many flights left" is treated as simply "healthy"


def clip_rul(df, cap=RUL_CAP):
    """Return a copy where every RUL label above `cap` is replaced by `cap`.

    Why: a healthy engine's sensors look the same at 300 flights left as at 150,
    so asking the model to tell those apart is asking the impossible. Clipping
    lets it say "healthy" and only count down once wear actually shows.
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_rolling_features(df, window):
    """Add one smoothed column per sensor: the average of its last `window` flights.

    Why: single readings are jumpy (noise). Averaging recent flights cancels the
    random wiggles and keeps the real trend.
    No peeking: flight 50's average uses flights 50, 49, 48... never flight 51,
    and it is computed per engine, so engines never mix. Early flights that don't
    have `window` flights behind them yet just average what exists so far.
    """
    df = df.copy()
    sensors = [c for c in feature_columns(df) if c.startswith("s")]
    by_engine = df.groupby("unit")
    for c in sensors:
        df[f"{c}_avg{window}"] = by_engine[c].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
    return df


def cross_validate_by_engine(make_model, df, cols, n_folds=5):
    """Average validation RMSE over `n_folds` different held-out engine groups.

    One 20-engine practice exam is noisy (Week 2). Here every engine takes a
    turn being held out, and we average the scores for a steadier number.
    Returns (mean RMSE, list of the per-fold RMSEs).
    """
    from sklearn.model_selection import GroupKFold

    scores = []
    for tr_idx, val_idx in GroupKFold(n_splits=n_folds).split(df, groups=df["unit"]):
        tr, val = df.iloc[tr_idx], df.iloc[val_idx]
        model = make_model().fit(tr[cols], tr["RUL"])
        scores.append(rmse(val["RUL"], model.predict(val[cols])))
    return float(np.mean(scores)), scores


def plot_scoreboard(scores, title):
    """Horizontal bar chart of test RMSE per model. `scores` = {model name: RMSE}.

    Shorter bar = smaller typical error = better. One color, because every bar
    measures the same thing.
    """
    import matplotlib.pyplot as plt

    names, values = list(scores.keys()), list(scores.values())
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.3))
    bars = ax.barh(names, values, color=ENGINE_COLORS[0], height=0.6)
    ax.invert_yaxis()  # first model at the top
    for bar, v in zip(bars, values):
        ax.text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=10)
    ax.set_xlabel("test RMSE: typical error in flights (shorter bar = better)")
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def stage_of_life(true_rul, cap=RUL_CAP):
    """Sort engines into three groups by how many flights they TRULY have left.

    Close to failure (30 or fewer), wearing (31 up to the cap), and healthy
    (more than the cap). Used only for grading, never as a clue for the model.
    """
    labels = ["close to failure (0-30)", f"wearing (31-{cap})", f"healthy (over {cap})"]
    return pd.cut(pd.Series(np.asarray(true_rul, dtype=float)),
                  bins=[-1, 30, cap, np.inf], labels=labels)


def plot_error_by_stage(true_rul, preds):
    """Grouped bars: RMSE of each model within each stage-of-life group.

    `preds` = {model name: predictions}. Shorter bar = better. Shows WHERE a
    model is good or bad, which a single overall RMSE hides.
    """
    import matplotlib.pyplot as plt

    truth = np.asarray(true_rul, dtype=float)
    stage = stage_of_life(truth)
    groups = list(stage.cat.categories)
    width = 0.8 / len(preds)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (name, pred) in enumerate(preds.items()):
        pred = np.asarray(pred, dtype=float)
        vals = [rmse(truth[(stage == g).to_numpy()], pred[(stage == g).to_numpy()])
                for g in groups]
        x = np.arange(len(groups)) + i * width
        bars = ax.bar(x, vals, width=width * 0.92, color=ENGINE_COLORS[i], label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}",
                    ha="center", fontsize=10)
    counts = [int((stage == g).sum()) for g in groups]
    ax.set_xticks(np.arange(len(groups)) + width * (len(preds) - 1) / 2)
    ax.set_xticklabels([f"{g}\n{n} engines" for g, n in zip(groups, counts)])
    ax.set_ylabel("RMSE: typical error in flights")
    ax.set_xlabel("how many flights the engine truly has left")
    ax.set_title("Where each model is wrong (100 test engines; shorter bar = better)", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return fig


def add_baseline_features(df, n_start=20, smooth=None):
    """Add one column per sensor: how far it has moved from THIS engine's own start.

    Why: every engine begins life at a slightly different level, like people with
    different resting heart rates. A reading that looks normal on the shared scale
    can be a big change for an engine that started unusually "healthy-looking".
    The starting level is the average of the engine's first `n_start` flights.
    No peeking: during those first flights we only average the flights so far.
    If `smooth` is given, also add the average of that change over the last
    `smooth` flights (past flights only), which steadies the jumpy readings.
    """
    df = df.copy()
    sensors = [c for c in feature_columns(df) if c.startswith("s")]
    by_engine = df.groupby("unit")
    for c in sensors:
        # Running average of the flights so far, frozen once n_start flights exist.
        so_far = by_engine[c].transform(lambda x: x.expanding().mean())
        frozen = by_engine[c].transform(lambda x: x.iloc[:n_start].mean())
        start_level = so_far.where(df["cycle"] <= n_start, frozen)
        df[f"{c}_chg"] = df[c] - start_level
    if smooth:
        by_engine = df.groupby("unit")
        for c in sensors:
            df[f"{c}_chg_avg{smooth}"] = by_engine[f"{c}_chg"].transform(
                lambda x: x.rolling(smooth, min_periods=1).mean())
    return df


def starting_score(df, n_start=20, reference=None):
    """One number per engine: how healthy its sensors look in its first flights.

    0 = an average engine; positive = starts out looking healthier than average.
    Built from the six most wear-sensitive sensors. `reference` is the table
    whose engines define "average" (always the training data).
    """
    reference = df if reference is None else reference
    corr = sensor_correlations(add_rul(reference) if "RUL" not in reference else reference)
    top = list(corr.index[:6])
    healthy_direction = np.sign(corr[top])   # +1 if a higher reading means healthier
    ref_start = reference[reference["cycle"] <= n_start].groupby("unit")[top].mean()
    start = df[df["cycle"] <= n_start].groupby("unit")[top].mean()
    z = (start - ref_start.mean()) / ref_start.std()   # distance from average, in spreads
    return (z * healthy_direction).mean(axis=1)


# ---------------------------------------------------------------------------
# Neural network helpers (the "Later" milestone: a 1D-CNN in PyTorch).
# PyTorch is imported inside each function, so the older scripts still run
# without it.
# ---------------------------------------------------------------------------

WINDOW = 30  # flights per "video clip". The shortest test engine has 31 flights, so every one fits.


def load_test_with_rul(add_features=None):
    """The FULL test history (every flight), with an RUL label on every row.

    The network needs each test engine's last 30 flights, not just its last row.
    RUL_FD001.txt gives the answer at the last flight; one flight earlier the
    engine had one more flight left, and so on back to the start.
    """
    test = load_raw("test_FD001.txt")
    if add_features is not None:
        test = add_features(test)
    true_rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["RUL"])["RUL"]
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    rul_at_end = test["unit"].map(dict(zip(range(1, len(true_rul) + 1), true_rul)))
    test["RUL"] = rul_at_end + (last_cycle - test["cycle"])
    return test


def fit_scaler(df, cols):
    """Average and spread of each column, measured on TRAINING rows only.

    Neural networks learn badly when one input is around 1,400 and another
    around 0.03. We shift and stretch every column to a similar range. The
    numbers come from training data only: measuring them on test engines too
    would be a small peek at the exam (data leakage).
    """
    mean = df[cols].mean()
    std = df[cols].std().replace(0, 1.0)  # guard against dividing by zero
    return mean, std


def apply_scaler(df, cols, scaler):
    df = df.copy()
    mean, std = scaler
    df[cols] = (df[cols] - mean) / std
    return df


def make_windows(df, cols, window=WINDOW, last_only=False):
    """Cut each engine's history into overlapping clips of `window` flights.

    Clip 1 = flights 1-30, clip 2 = flights 2-31, ... Each clip's label is the
    RUL at its LAST flight, so a clip only ever contains flights that have
    already happened. `last_only=True` keeps just the final clip per engine
    (used for grading test engines, like before).
    Returns X with shape (clips, window, features), y, and the engine of each clip.
    """
    X, y, units = [], [], []
    for unit, e in df.groupby("unit", sort=True):
        values = e[cols].to_numpy(dtype=np.float32)
        rul = e["RUL"].to_numpy(dtype=np.float32)
        ends = [len(e)] if last_only else range(window, len(e) + 1)
        for end in ends:
            if end < window:
                continue  # too short for a full clip (never happens in FD001)
            X.append(values[end - window:end])
            y.append(rul[end - 1])
            units.append(unit)
    return np.stack(X), np.array(y, dtype=np.float32), np.array(units)


def make_cnn(n_features, window=WINDOW):
    """A small 1D-CNN: two sliding-stencil layers, then a small decision layer.

    Conv1d = the stencil. kernel_size=5 means it looks at 5 neighboring flights
    at a time. The first layer has 16 stencils, the second 32; each learns to
    spot one kind of shape (a rise, a bend, a jump). Flatten + Linear then
    combine everything the stencils found into one number: the RUL guess.
    ReLU = "keep positive signals, zero out negative ones"; without it, stacking
    layers would collapse back into one straight-line model.
    """
    from torch import nn

    return nn.Sequential(
        nn.Conv1d(n_features, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * window, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


def predict_cnn(model, X, scale=RUL_CAP):
    """RUL guesses (in flights) for clips X of shape (clips, window, features)."""
    import torch

    model.eval()  # "exam mode": no learning happens here
    with torch.no_grad():
        Xt = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))
        return model(Xt).squeeze(1).numpy() * scale


def train_cnn(X, y, n_epochs=30, seed=0, X_val=None, y_val=None,
              lr=1e-3, batch_size=256, scale=RUL_CAP):
    """Train a fresh CNN on clips X with labels y. Returns (model, history).

    One epoch = the network sees every training clip once, in shuffled batches
    of 256. After each batch it nudges its numbers a little to shrink the error
    (that nudging is what "learning" means here). `history` records train and
    validation RMSE after every epoch, so we can watch for overfitting.
    Labels are divided by 125 while learning (networks prefer numbers near 0-1)
    and multiplied back for every prediction.
    """
    import torch

    torch.manual_seed(seed)  # same starting numbers every run = repeatable
    model = make_cnn(X.shape[2], X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))  # PyTorch wants (clips, features, flights)
    yt = torch.from_numpy(y / scale)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(n_epochs):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = ((model(Xt[idx]).squeeze(1) - yt[idx]) ** 2).mean()  # squared error, like RMSE
            loss.backward()   # work out which way to nudge each number
            opt.step()        # nudge them
        record = {"epoch": epoch + 1, "train": rmse(y, predict_cnn(model, X, scale))}
        if X_val is not None:
            record["val"] = rmse(y_val, predict_cnn(model, X_val, scale))
        history.append(record)
    return model, history
