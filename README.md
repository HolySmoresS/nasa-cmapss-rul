# NASA C-MAPSS Remaining Useful Life Prediction

**Goal:** predict how many flight cycles a jet engine has left before failure, using simulated NASA sensor data (C-MAPSS, subset FD001).

**Status:** Project 0, Week 1 done (data loaded, RUL labels, sensor plots, mean baseline).

## Results
Scored on the last recorded cycle of each of the 100 FD001 test engines. RMSE = typical error in cycles (lower is better).

| Model | Test RMSE (cycles) | Notes |
|---|---|---|
| Mean baseline | 52.62 | Predicts the training-average RUL (107.8) for every engine; ignores sensors |

## What wear looks like
![Six sensors for three engines, plotted against cycles until failure](figures/week1_sensors.png)

Sensors are flat and noisy while an engine is healthy, then bend sharply in roughly the last 100 cycles. Engines with very different lifetimes (128, 199, 362 cycles) follow the same shape near failure, so the sensors track *closeness to failure*, not age.

## What I'm learning
- **Labels:** training engines run to failure, so RUL = last cycle − current cycle.
- **Not every column is useful:** 7 columns never change (`op3, s1, s5, s10, s16, s18, s19`), leaving 17 usable features.
- **Baselines matter:** a "dumb" guess gives a number any real model must beat.
- **Data leakage:** the baseline uses training data only; test answers are used only for scoring.

## Project layout
| Path | What it is |
|---|---|
| `src/cmapss.py` | Shared helpers: loading data, RUL labels, RMSE, plots |
| `src/02_week1.py` | Week 1 as a script (prints results, saves the plot) |
| `notebooks/01_week1.ipynb` | Week 1 as a notebook, with explanations |
| `figures/` | Saved plots |
| `LOG.md` | Experiment log |

## Setup
```bash
python3 -m venv .venv                         # private package folder for this project
.venv/bin/pip install -r requirements.txt     # exact package versions used
.venv/bin/python src/02_week1.py              # run Week 1 as a script
.venv/bin/jupyter lab notebooks/01_week1.ipynb  # or open the notebook in your browser
```

## Data
NASA Prognostics Center of Excellence data repository, Turbofan Engine Degradation Simulation Data Set. Not included here; download it and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/`.
