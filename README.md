# NASA C-MAPSS Remaining Useful Life Prediction

**Goal:** predict how many flight cycles a jet engine has left before failure, using simulated NASA sensor data (C-MAPSS, subset FD001).

**Status:** Project 0, Week 3 done. Best model so far: gradient boosting with RUL clipped at 125, test RMSE 18.36.

## Results
Scored on the last recorded cycle of each of the 100 FD001 test engines. RMSE = typical error in cycles (lower is better).

| Model | Test RMSE (cycles) | Notes |
|---|---|---|
| Mean baseline | 52.62 | Predicts the training-average RUL (107.8) for every engine; ignores sensors |
| Linear regression, raw RUL labels | 32.04 | 17 features; validation RMSE 38 to 45 depending on which 20 engines are held out |
| Linear regression, RUL clipped at 125 | 21.90 | Same model; only the training labels changed |
| **Gradient boosting, RUL clipped at 125** | **18.36** | 100 small trees; studied engines 17.57 vs unseen 18.52, so no memorizing |
| Linear regression, clipped + rolling average (5 flights) | 21.79 | Rolling averages: no real gain |
| Gradient boosting, clipped + rolling average (5 flights) | 18.43 | Rolling averages: no real gain |

![Bar chart of test RMSE for every model](figures/week3_scoreboard.png)

Only training labels are clipped. Test engines are always graded against NASA's true RUL values.

## What wear looks like
![Six sensors for three engines, plotted against cycles until failure](figures/week1_sensors.png)

Sensors are flat and noisy while an engine is healthy, then bend sharply in roughly the last 100 cycles. Engines with very different lifetimes (128, 199, 362 cycles) follow the same shape near failure, so the sensors track *closeness to failure*, not age.

## Where linear regression goes wrong
![Predicted vs true RUL for the linear regression](figures/week2_predicted_vs_true.png)

The dots form a curve because a straight-line model cannot follow curved wear. It predicts impossible negative RUL near failure, flattens out around 150 to 190 for healthy engines, and guesses too high for 85 of the 100 test engines, which is the unsafe direction.

## What clipping and gradient boosting changed
![Raw vs clipped RUL labels for one engine](figures/week3_clipping.png)

A healthy engine's sensors look the same at 300 flights left as at 150. Clipping relabels everything above 125 as "healthy", so the model only counts down once wear shows.

![Predicted vs true RUL for three models](figures/week3_predicted_vs_true.png)

Left to right, the dots tighten around the dashed "perfect" line. With raw labels the model guessed too high for 85 of 100 test engines; with clipping, 58; with gradient boosting, 53.

## What I'm learning
- **Labels:** training engines run to failure, so RUL = last cycle − current cycle.
- **Not every column is useful:** 7 columns never change (`op3, s1, s5, s10, s16, s18, s19`), leaving 17 usable features.
- **Baselines matter:** a "dumb" guess gives a number any real model must beat.
- **Data leakage:** the baseline uses training data only; test answers are used only for scoring.
- **Validation by engine:** 20 of the 100 training engines are held out whole, so the model is checked on engines it has never seen.
- **Overfitting:** an unlimited decision tree scores 0.00 RMSE on the engines it studied and 54.91 on unseen ones.
- **Leakage in numbers:** if the model can see the engine ID, a by-row split reports 24.69 while the honest by-engine split reports 47.30.
- **Validation scores are noisy:** the same model scores 37.6 to 44.6 depending on which engines are held out.
- **Labels matter as much as models:** clipping RUL at 125 cut the error from 32.04 to 21.90 without touching the model.
- **Gradient boosting:** many small trees, each fixing the previous ones' mistakes, can follow curved wear that a straight line cannot.
- **Cross-validation:** averaging 5 practice exams (each engine held out once) gives a steadier score than one split.

## What failed
- **Week 1 plot bug:** failure showed on the wrong side because a shared axis was flipped six times.
- **Feature count:** my notes said 19 usable features; the data says 17.
- **First leakage demo:** adding only the flight number did not show leakage (by-row 52.60 vs by-engine 47.00). It only appears once the model can identify the engine.
- **Rolling-average features:** expected a clear gain, got none. Cross-validation RMSE moved from 18.57 to between 18.22 and 18.49 depending on the window, while the five folds themselves range from about 16 to 20, so the differences are noise. Test RMSE went from 18.36 to 18.43. The averages were checked by hand and are computed correctly. Likely reason (not proven): noise is not the main limit; engines differ from each other in ways smoothing cannot fix.

## Project layout
| Path | What it is |
|---|---|
| `src/cmapss.py` | Shared helpers: loading data, RUL labels, RMSE, plots |
| `src/02_week1.py`, `src/03_week2.py`, `src/04_week3.py` | Each week as a script (prints results, saves the plots) |
| `notebooks/01_week1.ipynb`, `02_week2.ipynb`, `03_week3.ipynb` | Each week as a notebook, with explanations |
| `figures/` | Saved plots |
| `LOG.md` | Experiment log |

## Setup
```bash
python3 -m venv .venv                         # private package folder for this project
.venv/bin/pip install -r requirements.txt     # exact package versions used
.venv/bin/python src/02_week1.py              # run Week 1 as a script
.venv/bin/python src/03_week2.py              # run Week 2 as a script
.venv/bin/python src/04_week3.py              # run Week 3 as a script (takes a few minutes)
.venv/bin/jupyter lab notebooks/01_week1.ipynb  # or open the notebook in your browser
```

## Data
NASA Prognostics Center of Excellence data repository, Turbofan Engine Degradation Simulation Data Set. Not included here; download it and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/`.
